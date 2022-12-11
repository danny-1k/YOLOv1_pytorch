import torch
import torch.nn as nn
from metrics import iou_t
from utils import convert_to_coords

class DetectionLoss(nn.Module):
    def __init__(self, S=7, B=2 , C=20, lambda_coord=5, lambda_noobj=.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss()

        # this only works if `self.B` = 2, I'm thinking of something that works for any B
        assert self.B == 2, "Only works for B = 2"


    def forward(self, pred, truth):
        # pred of shape N x S*S*(5*B + C)
        # reshape to size N x S x S x (5*B + C)

        pred = pred.view(-1, self.S, self.S, 5*self.B + self.C)

        # calculate the iou of the prediction against ground truth to determine the best of the B boxes 

        ious = iou_t(
            convert_to_coords(pred[..., :self.B*4].contiguous()).view(-1, 4),
            convert_to_coords(truth[..., :self.B*4].contiguous()).view(-1, 4),
        ).view(-1, self.S, self.S, self.B)

        iou_max, best_box = torch.max(ious, dim=-1)

        best_box = best_box.unsqueeze(-1)

        box_exists = truth[..., (4*self.B) + 1].unsqueeze(-1)


        # -----box coordinate loss------


        box_predictions = box_exists * (
            best_box * pred[..., 4:8]
            + (1 - best_box) * pred[..., 0:4]
        )

        box_targets = box_exists * truth[..., 0:4] # All the bounding boxes in truth are the same

        # Take the sqrt of the height's and width's so we can reduce the effect of MSE on smaller values
        sqrt_dims = torch.sign(box_predictions[..., 2:4]) *\
            torch.sqrt(
                torch.abs(box_predictions[..., 2:4].abs() + 1e-6)
            )

        box_predictions[..., 2:4] = sqrt_dims.clone().data#torch.rand_like(sqrt_dims)

        box_targets[..., 2:4] = torch.sqrt(
            box_targets[..., 2:4]
        )
        
        box_loss = self.mse(
            box_predictions.view(-1, 4),
            box_targets.view(-1, 4),
        )


        # -----object loss------

        pred_confidence = box_exists * (
            best_box * pred[..., 8:9]
            + (1 - best_box) * pred[..., 9:10]
        )

        target_confidence = box_exists * (
            truth[..., 8:9] # any one works
        )

        confidence_loss = self.mse(pred_confidence.view(-1, 1), target_confidence.view(-1, 1))


        # -----no object loss------

        pred_no_obj_1 = (1-box_exists) * pred[..., 8:9]
        pred_no_obj_2 = (1-box_exists) * pred[..., 9:10]

        truth_no_obj = (1-box_exists) * truth[..., 8:9]


        target_confidence = (1-box_exists) * (
            truth[..., 8:9] # any one works
        )

        no_object_loss = self.mse(
            pred_no_obj_1.view(-1, 1),
            truth_no_obj.view(-1, 1)
        ) \
        + self.mse(
            pred_no_obj_2.view(-1, 1),
            truth_no_obj.view(-1, 1)
        )


        # -----class loss------

        pred_class = box_exists * pred[..., -self.C:]
        truth_class = box_exists * truth[..., -self.C:]

        class_loss = self.mse(pred_class.view(-1, self.C), truth_class.view(-1, self.C))


        loss = self.lambda_coord*box_loss + confidence_loss + self.lambda_noobj*no_object_loss + class_loss # center loss and sqrt size loss

        return loss

       


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    p = torch.ones((1, 7, 7, 30), requires_grad=True)
    y = torch.ones((1, 7, 7, 30), requires_grad=True)/2

    # p[:, :, :, -20:] = torch.Tensor([2]*19 + [1])
    # y[:, :, :, -20:] = torch.Tensor([0]*19 + [1])


    loss = DetectionLoss()

    print(p.dtype, 'prediction')
    print(y.dtype, 'truth')

    l = loss(p, y)

    l.backward()