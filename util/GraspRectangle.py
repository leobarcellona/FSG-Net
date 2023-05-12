import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
#from util.peak import peak_local_max
from skimage.feature import peak_local_max
import cv2


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """
    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    def __len__(self):
        return len(self.grs)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_graspnet_file(cls, fname, friction=0.2, num_grasps = 100, camera = 'kinect',obj_id=None, min_center_dist=0):
        """
        Load grasp rectangles from a graspnet dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []

        rect_grasp = np.load(fname)
        rect_grasp = rect_grasp[rect_grasp[:,5]<friction]


        #extract num_grasps grasping rect for each object
        for obj in np.unique(rect_grasp[:,6]):

            #SE ABBIAMO SETTATO L'ID RECUPERA SOLO LE GRASP DI QUELL'ID ALTRIMETNI TUTTE
            if obj_id != None:
                if(int(obj+1) != obj_id):
                    continue

            obj_rect = rect_grasp[rect_grasp[:,6]==obj]

            step = 1
            if np.shape(obj_rect)[0] > num_grasps:
                step = int(np.shape(obj_rect)[0] / num_grasps)

            indexes = np.arange(0, np.shape(obj_rect)[0], step)

            obj_rect_small = obj_rect[indexes]

            #compute box corners 
            old_centers = []
            for i, rect in enumerate(obj_rect_small):
                center_x, center_y, open_x, open_y, height, score, object_id = rect
                center = np.array([center_x, center_y])

                exit = False
                for c in old_centers:
                    if np.linalg.norm(center-c) < min_center_dist:
                        exit = True
                        break

                if not exit:

                    old_centers.append(center)

                    left = np.array([open_x, open_y])
                    axis = left - center
                    normal = np.array([-axis[1], axis[0]])
                    normal = normal / np.linalg.norm(normal) * height / 2
                    p0 = center + normal + axis
                    p1 = center + normal - axis
                    p2 = center - normal - axis
                    p3 = center - normal + axis

                    p0 = (int(p0[1]),int(p0[0]))
                    p1 = (int(p1[1]),int(p1[0]))
                    p2 = (int(p2[1]),int(p2[0]))
                    p3 = (int(p3[1]),int(p3[0]))

                    gr = np.array([p0,p1,p2,p3])

                    '''
                    print("p0: ",p0," p1: ",p1,"p2: ",p2,"p3: ",p3)

                    img = cv2.circle(img, p0, radius=3, color=(255, 0, 255), thickness=-1)
                    img = cv2.circle(img, p1, radius=3, color=(255, 0, 255), thickness=-1)
                    img = cv2.circle(img, p2, radius=3, color=(255, 0, 255), thickness=-1)
                    img = cv2.circle(img, p3, radius=3, color=(255, 0, 255), thickness=-1)
                    '''
                    grs.append(GraspRectangle(gr))

        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta/180.0*np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
           if pad_to > len(self.grs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length/6, self.width/2).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi/6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """

        rr1, cc1 = self.polygon_coords()  #Generate coordinates of pixels within polygon
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1]) #Generate coordinates of pixels within polygon

        if abs((self.angle - gr.angle + np.pi/2) % np.pi - np.pi/2) > angle_threshold:
            #print("!!!!! Wrong angle")
            return 0, 0, 0, [None, rr1, cc1, rr2, cc2]

        #rr1, cc1 = self.polygon_coords()  #Generate coordinates of pixels within polygon
        #rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1]) #Generate coordinates of pixels within polygon

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            #print("!!!!! Exception")
            return 0, 0, 0, []

        canvas = np.zeros((r_max, c_max)) #sort of convex hall that contains the two bb
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0, 0, 0, [None, rr1, cc1, rr2, cc2]
        intersection = np.sum(canvas == 2)

        #return intersection/union
        #print("Here I am")
        return intersection/union, union, intersection, [canvas, rr1, cc1, rr2, cc2]

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None, label=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color, label=label)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """
    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
             [y1 - self.width/2 * xo, x1 - self.width/2 * yo],
             [y2 - self.width/2 * xo, x2 - self.width/2 * yo],
             [y2 + self.width/2 * xo, x2 + self.width/2 * yo],
             [y1 + self.width/2 * xo, x1 + self.width/2 * yo],
             ]
        ).astype(np.float))

    def max_iou(self, grs, overwrite_w=False):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        u_out, i_out = 0, 0
        draws_out = []
        gr_out = []

        #print("len GRS",grs.to_array().shape)
        for gr in grs:
            if overwrite_w:
                self.width = gr.width
                self_gr = self.as_gr
            iou, u, i, draws = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
            #print("+++ iou: {}, max iou: {}".format(iou,max_iou))
            if max_iou == iou:
                #print("+++ UPDATE LIST!")
                #print("union: {}, inter: {}".format(u,i))
                u_out, i_out =  u, i
                #print("draws: ", draws)
                draws_out = draws
                gr_out = gr

        #print("+++++ union: {}, inter: {}".format(u_out,i_out))
        #print("+++++ max_iou: {}".format(max_iou))
        #print("+++++ draws: ", draws_out)
        
        return max_iou, u_out, i_out, gr_out, draws_out

    def plot(self, ax, color=None, label=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color, label)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (self.center[1]*scale, self.center[0]*scale, -1*self.angle*180/np.pi, self.length*scale, self.width*scale)


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1, diff=0):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """

    #indexes of local maxima
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.0, num_peaks=no_grasps)

    #If no peaks get the center
    if len(local_max) < 1:
        print("No peaks")
        print(local_max)
        h = 360
        w = 360
        local_max = np.expand_dims(np.array((int(h / 2), int(w / 2)), np.int), axis=0)
        print(local_max)
        print("Changed peaks")

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        grasp_point_2 = list(grasp_point)
        grasp_point_2[1] += diff
        grasp_point_2 = tuple(grasp_point_2)

        g = Grasp(grasp_point_2, grasp_angle)
        if width_img is not None:

            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps.append(g)

    return grasps


def detect_grasps2(q_img, ang, width=None, no_grasps=1, diff=0):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang: Angle value
    :param width: (optional) Width value
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """

    #indexes of local maxima

    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.0, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:

        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang

        grasp_point_2 = list(grasp_point)
        grasp_point_2[1] += diff
        grasp_point_2 = tuple(grasp_point_2)
        ##
        #print(grasp_point_2)
        g = Grasp(grasp_point_2, grasp_angle)
        if width is not None:

            g.length = width
            g.width = g.length/2

        grasps.append(g)

    return grasps

