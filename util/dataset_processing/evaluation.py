import numpy as np
import matplotlib.pyplot as plt

from .. import GraspRectangle
from ..GraspRectangle import GraspRectangles


def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = GraspRectangle.detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, diff=0, outputs=False,
                        propagate=False, seg_map=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangle.GraspRectangles):
        gt_bbs = GraspRectangle.GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs

    #print("gt_bbs: ", len(gt_bbs))

    gs = GraspRectangle.detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps, diff=diff)
    max = 0
    max_gr = []
    max_u, max_i = 0, 0
    max_draws = []
    max_gt_bb = []
    #print("ciccio banano")
    if seg_map is None:
        #print("gs len: ",len(gs))
        for g in gs:

            iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=False)

            #print("iou: ", iou)
            #print("gt_bb: ", gt_bb)

            if max<=iou:
                max = iou
                max_gr = g
                max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

    else:
        # Se abbiamo passato la maschera di segmentazione

        # Settimao la prima grasp come la migliore
        iou, u, i, gt_bb, draws = gs[0].max_iou(gt_bbs, overwrite_w=False)
        max = iou
        max_gr = gs[0]
        max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

        center = gs[0].as_gr.center
        max_activation_seg = seg_map[0, 1, int(center[0]), int(center[1]-diff)]
        #print(max_activation_seg)

        for g in gs:
            # per ogni grasp prendiamo quella con segmentazione più alta
            center = g.as_gr.center
            activation_seg = seg_map[0, 1, int(center[0]), int(center[1]-diff)]
            #print(activation_seg)
            if max_activation_seg < activation_seg:
                iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=True)

                max = iou
                max_gr = g
                max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb
                max_activation_seg = activation_seg

            elif max_activation_seg == activation_seg:
                iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=True)
                if max <= iou:
                    max = iou
                    max_gr = g
                    max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

            #print(max_activation_seg)

    if outputs:

        shape = (721, 1281)
        shape2 = (721, 721)
        f = plt.figure(1)
        ax1 = f.add_subplot(1, 2, 1)
        ax2 = f.add_subplot(1, 2, 2)
        ax1.imshow(np.zeros(shape))
        ax1.axis([0, shape[1], shape[0], 0])
        ax2.imshow(np.zeros(shape))
        ax2.axis([0, shape[1], shape[0], 0])

        print("Max IoU: ", max)
        print("Inter={}, Union={}".format(max_i,max_u))
        print("GT BB length: {}, GT BB width: {}, RATIO: {}".format(max_gt_bb.length, max_gt_bb.width, max_gt_bb.length/max_gt_bb.width))

        canva, rr1, cc1, rr2, cc2 = max_draws
        img = np.zeros(shape)
        img[rr1, cc1] += 50
        img[rr2, cc2] += 50
        ax1.imshow(img)
        #max_gr.plot(ax=ax1)
        max_gr.plot(ax=ax2, color=(0,1.0,0.0,1.0), label="GR from heatmap") #green = BB computed from heatmap
        max_gt_bb.plot(ax=ax2, color=(1.0,0.0,0.0,1.0), label="Ground Truth GR") # red = GT BB
        ax2.legend()
        plt.show()

    if propagate:
        #print("max:", max)
        #print("max_draws: ",max_draws)
        return max, max_i, max_u, max_draws, max_gt_bb, max_gr

    if max>0.25 : #IMHO --> >=0.45 sarebbero buoni 
        return True
    else:
        return False


def calculate_iou_match2(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, diff=0, outputs=False,
                    propagate=False, seg_map=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangle.GraspRectangles):
        gt_bbs = GraspRectangle.GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs

    gs = GraspRectangle.detect_grasps2(grasp_q, grasp_angle, width=grasp_width, no_grasps=no_grasps, diff=diff)
    max = 0
    max_gr = []
    max_u, max_i = 0, 0
    max_draws = []
    max_gt_bb = []
    if seg_map is None:
        for g in gs:

            iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=True)

            if max<=iou:
                max = iou
                max_gr = g
                max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

    else:
        # Se abbiamo passato la maschera di segmentazione

        # Settimao la prima grasp come la migliore
        iou, u, i, gt_bb, draws = gs[0].max_iou(gt_bbs, overwrite_w=True)
        max = iou
        max_gr = gs[0]
        max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

        center = gs[0].as_gr.center
        max_activation_seg = seg_map[0, 1, int(center[0]), int(center[1]-diff)]
        #print(max_activation_seg)

        for g in gs:
            # per ogni grasp prendiamo quella con segmentazione più alta
            center = g.as_gr.center
            activation_seg = seg_map[0, 1, int(center[0]), int(center[1]-diff)]
            #print(activation_seg)
            if max_activation_seg < activation_seg:
                iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=True)

                max = iou
                max_gr = g
                max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb
                max_activation_seg = activation_seg

            elif max_activation_seg == activation_seg:
                iou, u, i, gt_bb, draws = g.max_iou(gt_bbs, overwrite_w=True)
                if max <= iou:
                    max = iou
                    max_gr = g
                    max_i, max_u, max_draws, max_gt_bb = i, u, draws, gt_bb

            #print(max_activation_seg)

    if outputs:

        shape = (721, 1281)
        shape2 = (721, 721)
        f = plt.figure(1)
        ax1 = f.add_subplot(1, 2, 1)
        ax2 = f.add_subplot(1, 2, 2)
        ax1.imshow(np.zeros(shape))
        ax1.axis([0, shape[1], shape[0], 0])
        ax2.imshow(np.zeros(shape))
        ax2.axis([0, shape[1], shape[0], 0])

        print("Max IoU: ", max)
        print("Inter={}, Union={}".format(max_i,max_u))
        print("GT BB length: {}, GT BB width: {}, RATIO: {}".format(max_gt_bb.length, max_gt_bb.width, max_gt_bb.length/max_gt_bb.width))

        canva, rr1, cc1, rr2, cc2 = max_draws
        img = np.zeros(shape)
        img[rr1, cc1] += 50
        img[rr2, cc2] += 50
        ax1.imshow(img)
        #max_gr.plot(ax=ax1)
        max_gr.plot(ax=ax2, color=(0,1.0,0.0,1.0), label="GR from heatmap") #green = BB computed from heatmap
        max_gt_bb.plot(ax=ax2, color=(1.0,0.0,0.0,1.0), label="Ground Truth GR") # red = GT BB
        ax2.legend()
        plt.show()

    if propagate:
        #print("max:", max)
        #print("max_draws: ",max_draws)
        return max, max_i, max_u, max_draws, max_gt_bb, max_gr

    if max>0.25 : #IMHO --> >=0.45 sarebbero buoni 
        return True
    else:
        return False