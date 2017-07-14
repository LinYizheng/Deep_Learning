import numpy as np
import cv2

robot2world_offset_mm = np.array([0, 0, 630], dtype=float)
peg2robot_offset_mm = np.array([105, 0, 0], dtype=float)
peg_points_mm = np.array([[   0,   0,   0],
                          [ 100,  30,  30],
                          [ 100, -30,  30],
                          [ 100,  30, -30],
                          [ 100, -30, -30],
                          [-100,  30,  30],
                          [-100, -30,  30],
                          [-100,  30, -30],
                          [-100, -30, -30]], dtype=float)
brick_points_mm = np.array([[   0,    0, 50],
                            [  40,   40, 50],
                            [  40,  -40, 50],
                            [ -40,   40, 50],
                            [ -40,  -40, 50],
                            [ 160,  160, 50],
                            [-160,  160, 50],
                            [ 160, -160, 50],
                            [-160, -160, 50]], dtype=float)
fov = 1.0808390005411683

def robot2world_m(robot_coord_mm):
    return (robot_coord_mm + robot2world_offset_mm) / 1000


def world2robot_mm(world_coord_m):
    return world_coord_m * 1000 - robot2world_offset_mm

def get_peg_coord_in_robot_mm(ee_pos_mm, ee_rot_rad):
    pos = ee_pos_mm + peg2robot_offset_mm
    rot = ee_rot_rad
    rm = rot_mat(rot)
    return rm.dot(peg_points_mm.T).T + pos


def get_brick_coord_in_world_m(center_pos):
    pos = center_pos[:3] * 1000
    rot = np.array([center_pos[5], center_pos[4], center_pos[3]])

    Tow = transformation_matrix(np.concatenate((pos, rot)), False)
    Pxw = Tow.dot(np.c_[brick_points_mm, np.ones(brick_points_mm.shape[0])].T)
    return Pxw[:3].T / 1000


def rot_mat(rot):
    u, v, w = rot
    Rx = np.array([[1, 0, 0],
                  [0, np.cos(w), -np.sin(w)],
                  [0, np.sin(w), np.cos(w)]], dtype=np.float32)
    Ry = np.array([[np.cos(v), 0, np.sin(v)],
                  [0, 1, 0],
                  [-np.sin(v), 0, np.cos(v)]], dtype=np.float32)
    Rz = np.array([[np.cos(u), -np.sin(u), 0],
                  [np.sin(u), np.cos(u), 0],
                  [0, 0, 1]], dtype=np.float32)

    return Rz.dot(Ry.dot(Rx))


def transformation_matrix(pose, degree):
    position = pose[:3]
    rot = pose[3:]
    if degree:
        rot /= 180.0 / np.pi
    rotMat = rot_mat(rot)
    tfMat = np.eye(4, dtype=np.float32)
    tfMat[:3, :3] = rotMat
    tfMat[:3, -1] = position
    return tfMat


def projectPtsToImg(points, camera_pose, img_size, degree=False):
    f = img_size / (np.tan(fov / 2.0) * 2.0)
    cameraMatrix = np.array([
        [f, 0, img_size / 2.0],
        [0, f, img_size / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    Tcw = transformation_matrix(camera_pose, degree)
    Twc = np.linalg.inv(Tcw)

    Pxw = np.pad(points.T, ((0, 1), (0, 0)), 'constant', constant_values=1)

    Pxc = Twc.dot(Pxw)[:3]
    scaled_img_points = cameraMatrix.dot(Pxc)
    img_points = scaled_img_points[:2] / scaled_img_points[2]

    return img_points.T.reshape(points.shape[0], -1, 2)


def get3DPtsFromImg(points, zw, camera_pose, img_size, degree=False):
    f = img_size / (np.tan(fov / 2.0) * 2.0)
    cameraMatrix = np.array([
        [f, 0, img_size / 2.0],
        [0, f, img_size / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    inv_cameraMatrix = np.linalg.inv(cameraMatrix)

    Tcw = transformation_matrix(camera_pose, degree)

    img_points = np.pad(points.T, ((0, 1), (0, 0)), 'constant', constant_values=1)
    Pxc = inv_cameraMatrix.dot(img_points)
    Pxc = np.pad(Pxc, ((0, 1), (0, 0)), 'constant', constant_values=1)
    Pxw = Tcw.dot(Pxc)

    camera_origin = camera_pose[:3].reshape(3, 1)
    space_points = (Pxw[:2] - camera_origin[:2]) / (Pxw[2] - camera_origin[2]) * (zw - camera_origin[2]) + camera_origin[:2]
    return space_points.T

if __name__ == '__main__':
    from pi_robot_API import Communication
    import time
    imgsz = 480
    pointA = 1
    pointB = 5

    camera_pose = np.array([1420, -450, 1180, 1.08, 0.003, -1.77])

    com = Communication()  # pi ros communication

    # brick_pos = np.array([0.6, 0, 0.775, 0, 0, -np.pi/6])
    brick_pos = np.array([ 0.6130, -0.1270,  0.7750,  0.0000,  0.0000,  0/180*np.pi])
    com.Set_Object_Pos('hole', brick_pos)
    time.sleep(1)

    brick_pos = np.concatenate((brick_pos[:3]*1000, np.array([brick_pos[5], brick_pos[4], brick_pos[3]])))
    Tow = transformation_matrix(brick_pos, False)
    Pxw = Tow.dot(np.c_[brick_points_mm, np.ones(brick_points_mm.shape[0])].T)
    points = Pxw[:3, 1:].T
    print points
    # peg_coords = get_peg_coord_in_robot_mm(np.array([418.4, 0, 629.89]), np.array([0, -0.5*np.pi, -np.pi]))
    # points = robot2world_m(peg_coords) * 1000

    image = com.Get_image_RGB()
    image = cv2.resize(image, (imgsz, imgsz))

    ProjImgPts = projectPtsToImg(points, camera_pose, imgsz)
    print get3DPtsFromImg(ProjImgPts[:, 0], 825, camera_pose, imgsz)

    real_distance = np.linalg.norm(brick_points_mm[pointA] - brick_points_mm[pointB])
    img_distance = np.linalg.norm(ProjImgPts[pointA] - ProjImgPts[pointB])
    print 'real distance', real_distance
    print 'image distance', img_distance
    print 'real distance per pixel', real_distance / img_distance

    n = 0
    for coord in ProjImgPts[:, 0]:
        if n == pointA or n == pointB:
            color = np.array([0, 0, 255, 255])
        else:
            color = np.array([0, 0, 0, 255])
        cv2.circle(image, tuple(np.round(coord).astype(int)), radius=3, color=color, thickness=2)
        n += 1
    cv2.imshow('image', image)
    cv2.waitKey(0)
