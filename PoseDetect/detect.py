# Detecting faces, landmarks and transforms

import dlib, cv2
import numpy as np

JAW_IDX = list(np.arange(0, 17))
FACE_IDX = list(np.arange(17, 68))
MOUTH_IDX = list(np.arange(48, 61))

RIGHT_EYE_IDX = list(np.arange(36, 42))
LEFT_EYE_IDX = list(np.arange(42, 48))

NOSE_IDX = list(np.arange(27, 36))
LEFT_EYE_BROW_IDX = list(np.arange(22, 27))
RIGHT_EYE_BROW_IDX = list(np.arange(17, 22))

FACTOR_Rm = 0.5
FACTOR_Rn = 0.5

dlib_path = '/data/liubo/face/annotate_face_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)


def faces(img):
    """
    Function to return faces detected in the image

    :param array img: Input image

    :return array: Array of rectangles ( dlib )
    """
    return detector(img, 1)


def landmarks(img, roi):
    """
    Function to return facial landmarks in the ROI provided

    :param array img: Input image
    :param roi: ROI ( Dlib rectangle ) like: (x1, y1, x2, y2)

    :return: Shape object with coordinates of landmarks
    """
    shape = predictor(img, roi)
    return shape


def face_pose(img, roi, pts):
    """
    Function to compute face pose given an ROI

    :param array img: Input image
    :param roi: ROI ( Dlib rectangle ) like: (x1, y1, x2, y2)
    :param array pts: Shape object with coordinates of landmarks [ Numpy array ]

    :return tuple: (normal, yaw ( in radians ), pitch ( in radians )) of the given face in the ROI
    """

    # Computing required points
    mid_eye = (pts[36] + pts[39] + pts[42] + pts[45])/4.0
    face_c = np.array([roi.left() + roi.right(), roi.top() + roi.bottom()])/2.0
    nose_tip = pts[30]
    mouth_c = (pts[48] + pts[54])/2.0
    nose_base = pts[33]

    # print nose_base, nose_tip

    # Computing required distances
    mid_eye_mouth_d = np.linalg.norm(mid_eye - mouth_c)
    nose_base_tip_d = np.linalg.norm(nose_base - nose_tip)

    # print mid_eye_mouth_d, nose_base_tip_d

    def find_sigma(d1, d2, Rn, theta):
        dz = 0
        m1 = (d1*d1)/(d2*d2)
        m2 = np.cos(theta)**2
        Rn2 = Rn**2
        if m2 == 1:
            dz = np.sqrt(Rn2/( m1 + Rn2 ))
        else:
            dz = np.sqrt(((Rn2)-m1-2*m2*(Rn2) + np.sqrt(((m1-(Rn2))*(m1-(Rn2))) + 4*m1*m2*(Rn2)))/(2*(1-m2)*(Rn2)))
        # print dz
        return np.arccos(dz)

    # Computing required angles
    t = mid_eye - nose_base
    symm_x = np.pi - np.arctan2(t[1], t[0])
    t = nose_tip - nose_base
    tau = np.pi - np.arctan2(t[1], t[0])
    theta = np.abs(tau - symm_x)
    sigma = find_sigma(nose_base_tip_d, mid_eye_mouth_d, FACTOR_Rn, theta)

    # print np.degrees(symm_x), np.degrees(tau), theta, sigma

    # Computing face pose
    normal = np.zeros(3)
    # print sigma, tau, theta, symm_x
    sin_sigma = np.sin(sigma)
    normal[0] = sin_sigma*np.cos(tau)
    normal[1] = -sin_sigma*np.sin(tau)
    normal[2] = -np.cos(sigma)

    # print normal

    n02 = normal[0]**2
    n12 = normal[1]**2
    n22 = normal[2]**2

    pitch = np.arccos(np.sqrt((n02 + n22)/(n02 + n12 + n22)))
    if nose_tip[1] - nose_base[1] < 0:
        pitch = -pitch

    yaw = np.arccos(np.abs(normal[2]/np.linalg.norm(normal)))
    if nose_tip[0] - nose_base[0] < 0:
        yaw = -yaw

    # print yaw, pitch
    return normal, yaw, pitch


def frontal_transform(pts, normal, yaw, pitch):
    """
    Function to return the frontal transform of the keypoints ( uses roll angle )

    :param array pts: Shape object with coordinates of landmarks [ Numpy array ]
    :param array normal: Array of length=3, describing the facial normal
    :param float yaw: Yaw angle of the face in the ROI ( in radians )
    :param float pitch: Pitch angle of the face in the ROI ( in radians )

    :return array: Frontal transform of the facial keypoints
    """
    def predict_z(pts, normal):
        pts_new = []
        for pt in pts:
            z = pt[0]*normal[0] + pt[1]*normal[1]
            z /= (-1)*(normal[2])
            pts_new.append(np.array([pt[0], pt[1], z]))
        return pts_new

    def rotation_transform(pts, R):
        ret = []
        # print R
        for pt in pts:
            # print pt
            ret.append(np.dot(R, pt))
        return np.array(ret)

    def get_rotation_matrix(axis, theta):
        # axis: 0 for 'X', 1 for 'Y', 2 for 'Z'
        C, S = np.cos(theta), np.sin(theta)
        R = None
        if axis == 0:
            R = np.array([[1.0, 0, 0],
                          [0, C, -S],
                          [0, S, C]] )
        elif axis == 1:
            R = np.array([[C, 0, S],
                          [0, 1.0, 0],
                          [-S, 0, C]] )
        else:
            R = np.array([[C, -S, 0],
                          [S, C, 0],
                          [0, 0, 1.0]] )
        return R

    def compute_roll(pts):
        # pts: Yaw, pitch corrected points
        # p = pts[29] - pts[27]
        p1 = pts[36] + pts[39]
        p2 = pts[42] + pts[45]
        p1 /= 2.0
        p2 /= 2.0
        p = p1 - p2
        t = np.arctan2(p[0], p[1])
        t = t+np.pi/2
        # print np.degrees(t)
        return t

    pts_ = predict_z(pts, normal)
    # print pts_

    R_yaw = get_rotation_matrix(1, yaw)
    R_pitch = get_rotation_matrix(0, pitch)

    # print R_yaw
    pts1 = rotation_transform(pts_, R_yaw)
    pts2 = rotation_transform(pts1, R_pitch)

    roll = compute_roll(pts2)
    R_roll = get_rotation_matrix(2, roll)
    pts3 = rotation_transform(pts2, R_roll)

    # print pts3

    return pts3
