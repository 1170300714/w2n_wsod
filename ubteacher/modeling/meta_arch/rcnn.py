# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from numpy.core.defchararray import decode
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import numpy as  np
import torch

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # print(proposals_rpn)

            # exit()
        


            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        elif branch == 'jitter_box':
   
            per_gt_instance = gt_instances[0]
            image_height,image_width = per_gt_instance.image_size
           
            gt_boxes = per_gt_instance.gt_boxes.tensor
            gt_scores = per_gt_instance.scores
            gt_classes = per_gt_instance.gt_classes
          
            
            N=10
            
            ans_box = []
            ans_class = []
            ans_scores = []
            for i in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[i].clone().cpu().numpy()
                gt_box_width = gt_box[2]-gt_box[0]
                gt_box_height = gt_box[3]-gt_box[1]
                jitter_boxes =torch.Tensor(N,4)
                for j in range(N):
                    offset_1 = np.random.uniform(-0.06,0.06)
                    
                    offset_height = gt_box_height*offset_1

                    offset_2 = np.random.uniform(-0.06,0.06)
                    
                    offset_width = gt_box_width*offset_2

                    jitter_boxes[j,0] = min(max(0,gt_box[0]+offset_width),image_width)
                    jitter_boxes[j,2] = min(max(0,gt_box[2]+offset_width),image_width)
                    jitter_boxes[j,1] = min(max(0,gt_box[1]+offset_height),image_height)
                    jitter_boxes[j,3] = min(max(0,gt_box[3]+offset_height),image_height)
          
                jitter_boxes = jitter_boxes.to(gt_boxes[i].device)
                proposals_ins_jitter = Instances((image_height,image_width))
                proposals_ins_jitter.proposal_boxes = Boxes(jitter_boxes)
                proposals_ins_jitter.objectness_logits = torch.ones((10,))

                gt_ins = Instances((image_height,image_width))

                gt_ins.gt_boxes = Boxes(gt_boxes[i].view(1,-1))
               
                gt_ins.gt_classes = gt_classes[i].view(1)
                gt_ins.scores = gt_scores[i].view(1)
              
              
                _,predictions  = self.roi_heads(
                    images,
                    features,
                    [proposals_ins_jitter],
                    [gt_ins],
                    compute_loss=False,
                    branch="jitter_box",
                    compute_val_loss=False,
                )
                
                bbr_reg_ans = predictions[1][:,gt_classes[i]*4:(gt_classes[i]+1)*4]
                decode_box = self.roi_heads.box_predictor.box2box_transform.apply_deltas(bbr_reg_ans,jitter_boxes)
             
                sigma = decode_box.std(axis=0).sum()/(2*(gt_box_width+gt_box_height))
                
               
              
                if sigma<0.02:
                    ans_box.append(gt_box)
                    ans_class.append(gt_classes[i].clone().cpu().numpy())
                    ans_scores.append(gt_scores[i].clone().cpu().numpy())


              
                
            if len(ans_box)>0:
                ans_int = Instances((image_height,image_width))
                ans_int.gt_boxes = Boxes(torch.from_numpy(np.array(ans_box)).to(gt_boxes[i].device))
                ans_int.gt_classes = torch.from_numpy(np.array(ans_class)).to(gt_boxes[i].device)
                ans_int.gt_scores = torch.from_numpy(np.array(ans_scores)).to(gt_boxes[i].device)
                return [ans_int]

                # ans_int.
        
            return None
            

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
