# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

def tailr_loss(lprobs, target, epsilon, min_weight, gamma, probs_model, ignore_index=None, reduce=True):
    lprobs = torch.nn.functional.softmax(lprobs, dim=-1).log()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    weight_theta_hat = probs_model.gather(dim=-1, index=target)
    

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        raise NotImplementedError

    
    with torch.no_grad():
        weight_theta_hat = (weight_theta_hat.log() - (gamma + (1 - gamma) * weight_theta_hat).log()).exp()
        weight_theta_hat = torch.clamp(weight_theta_hat, min=min_weight, max=1.0) 
    

    tailr_loss = weight_theta_hat * nll_loss
    # Can also adjust smooth loss accordingly; but no big impact
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        tailr_loss = tailr_loss.sum()
    eps_i = epsilon / (lprobs.size(-1))
    loss = (1. - epsilon) * tailr_loss + eps_i * smooth_loss
    return loss, nll_loss




@register_criterion(
    "tailr"
)
class TaiLr(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        density_min_weight,
        density_ratio_threshold,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.min_weight = density_min_weight
        self.gamma = density_ratio_threshold
    
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", default=0.0, type=float,
                            help="epsilon for label smoothing, 0 means no label smoothing")
        parser.add_argument("--report-accuracy", action="store_true",
                            help="report accuracy metric")
        parser.add_argument("--sentence-level", action="store_true")
        parser.add_argument("--ignore-prefix-size", default=0, type=int,
                            help="Ignore first N tokens")
        parser.add_argument("--density-min-weight", default=0.0, type=float)
        parser.add_argument("--density-ratio-threshold", default=1.0, type=float)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        

        net_output = model(**sample["net_input"])
        loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if reduce:
            return loss, sample_size, logging_output

        else:
            logging_output["nll_loss"] = logging_output["nll_loss"].view(lprobs.size(0), lprobs.size(1))
            return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        probs_model = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1))

        loss, nll_loss = tailr_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            min_weight=self.min_weight,
            gamma=self.gamma,
            probs_model=probs_model
        )
        return loss, nll_loss, lprobs

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
