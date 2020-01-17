
import numpy as np
import open3d as o3d

def points2PointCloud(points):
    """ Convert numpy point cloud to Open3D point cloud
    """
    if isinstance(points, o3d.geometry.PointCloud):
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[..., 0:3])
    if points.shape[-1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(points[..., 3:6])
    return pcd


def pointCloud2Points(pointCloud):
    """ Convert numpy point cloud to Open3D point cloud
    """
    pc = np.asarray(pointCloud.points)

    # Color
    if len(pointCloud.colors) == len(pointCloud.points):
        pc = np.hstack((pc, np.asarray(pointCloud.colors)))

    return pc


def crop_bbox(pc, bbox, return_outliers=False):
    """Crop the point cloud to given bounding box.

    Parameters
    ----------
    pc : numpy.ndarray
        Nx3 point cloud
    bbox : list or numpy.ndarray
        6 element list of bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]
    return_outliers : bool, optional
        If True, return the outlier indices too, by default False

    Returns
    -------
    list or tuple
        If return_outliers is False, indices of inliers returns
        If False, indices of inliers and indices of outliers returns
    """

    bbox = np.array(bbox).reshape((3, 2))
    filt = np.logical_and.reduce((
        pc[:, 0] >= bbox[0, 0],
        pc[:, 0] <= bbox[0, 1],
        pc[:, 1] >= bbox[1, 0],
        pc[:, 1] <= bbox[1, 1],
        pc[:, 2] >= bbox[2, 0],
        pc[:, 2] <= bbox[2, 1],
    ))
    inliers = np.where(filt)[0]
    outliers = np.where(np.logical_not(filt))[0]

    if return_outliers:
        return inliers, outliers

    return inliers


def plane_fit(points):
    """
    p, n = plane_fit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = points.T
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1],\
    "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, np.linalg.svd(M)[0][:, -1]


def get_rotation_matrix(init, target):
    """Get rotation matrix that rotates the unit vector init to unit vector target
    https://en.wikipedia.org/wiki/Rotation_matrix

    Parameters
    ----------
    init : list or numpy array
        Initial unit vector
    target list or numpy array
        Target unit vector

    Returns
    -------
    numpy array
        3x3 rotation matrix
    """

    a = init
    b = target

    v = np.cross(a, b)       # Rotation axis
    s = np.linalg.norm(v)    # sine of angle
    c = np.dot(a, b)         # cosine of angle

    v_x = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = c*np.eye(3) + s*v_x + (1-c) * np.outer(v, v)

    return R



def cut_pc(pc, line, keep=False):
    """Cut pc from given line equation (+y, forward direction)

    Parameters
    ----------
    pc : numpy array
        Input pont cloud
    line : list
        Line equation parameters [a, b] for y = a*x+b
    keep : bool, optional
        Whether to keep filtered out points as [0,0,0] or not, by default False

    Returns
    -------
    numpy array
        Output point cloud
    """
    '''
    pc: Point cloud
    line: [a,b] for y = ax + b
    keep: whether to keep filtered the points as zeros
    '''
    a, b = line
    filt = pc[:, 1] > (a*pc[:, 0] + b)
    if keep:
        pc[np.logical_not(filt), :3] = [0, 0, 0]
    else:
        pc = pc[filt]
    return pc

def transform(pc, transformation):
    """Apply Euclidean transformation to point cloud

    Parameters
    ----------
    pc : numpy array
        point cloud
    transformation : numpy array
        4x4 transormation matrix

    Returns
    -------
    numpy array
        transformed point cloud
    """

    pc_T = np.hstack((pc[:, :3], np.ones((len(pc), 1)))).T
    pc[:, :3] = np.dot(transformation, pc_T)[:3].T

    return pc


def rot_matrix(degree, axis='z'):
    '''
    Create 3x3 rotation matrix (default around z axis)
    '''
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)

    if axis == 'x':
        return np.array(((1,0,0), (0,c,-s), (0,s,c)))
    elif axis == 'y':
        return np.array(((c,0,-s), (0,1,0), (s,0,c)))
    elif axis == 'z':
        return np.array(((c,-s,0), (s,c,0), (0,0,1)))

def rotate(vect, degree, axis='z', about=[0,0,0]):
    '''
    Rotate a vector or set of vectors around x, y, or z axis about a point
    Inputs:
        vect: Nx3
    '''

    init_shape = vect.shape
    if len(init_shape) == 1:
        vect = vect.reshape((1, -1))

    about = np.reshape(about, (1, 3))
    R = rot_matrix(degree, axis=axis)

    pts = vect[:, :3]
    pts -= about
    pts = R.dot(pts.T).T
    pts += about
    vect[:, :3] = pts

    return vect

def get_normals(points, knn=None, radius=None):
    """Calculate normals of the given point cloud
    If only radius is given, search method is radius based.
    If only knn is given, search method is KNN based.
    If both are given, search method is hybrid.
    
    Arguments:
        points {array or PointCloud} -- Input point cloud
    
    Keyword Arguments:
        knn {int} -- K of K-Nearest Neighbor algorithm (default: {None})
        radius {float} -- Search radius (default: {None})
    
    Returns:
        array or PointCloud -- Output is array of normals if the input is points array,
        or it is PointCloud with normals if the input is PointCloud.
    """
    assert not (radius is None and knn is None)
    if knn is None:
        method = o3d.geometry.KDTreeSearchParamRadius(radius=radius)
    elif radius is None:
        method = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    else:
        method = o3d.geometry.KDTreeSearchParamHybrid(max_nn=knn, radius=radius)
    
    pcd = points2PointCloud(points)
    pcd.estimate_normals(method)

    if isinstance(points, o3d.geometry.PointCloud):
        return pcd
    
    return np.asarray(pcd.normals)


def euclidean_clustering(points, threshold, search_size, valid_indices=None, size_threshold=None, return_outliers=False, condition=None):
    """Returns list of clusters
    TODO: Update code
    """
    if valid_indices is None:
        pcd = points2PointCloud(points)
    else:
        pcd = points2PointCloud(points[valid_indices])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    P = np.asarray(pcd.points)
    P_unprocessed = np.ones((len(P),), dtype=np.bool)

    empty_list_template = np.zeros((len(P),), dtype=np.bool)

    clusters = []
    while P_unprocessed.sum() > search_size:

        Q = empty_list_template.copy()
        p_i = np.where(P_unprocessed)[0][0] # Get the next unprocessed index
        Q[p_i] = True

        while True:
            unprocessed_Q = np.logical_and(Q, P_unprocessed)
            if not unprocessed_Q.any():
                break

            q_i = np.where(unprocessed_Q)[0][0]  # Get the next unprocessed index in the queue
            P_unprocessed[q_i] = False  # Set the index as processed

            # Search neighbors of the currently processing index
            k, neig, _ = pcd_tree.search_radius_vector_3d(P[q_i], threshold)
            if condition is not None:
                neig = condition(pcd, q_i, neig)
            Q[neig] = True

        # Append the list of indices of the queue into clusters
        clusters.append(np.where(Q)[0])

    # Remove cluster whose sizes are smaller than the size_threshold
    if size_threshold:
        for i in np.arange(len(clusters))[::-1]:
            c = clusters[i]
            if len(c) < size_threshold:
                P_unprocessed[c] = True
                del clusters[i]

    # Fix cluster indices
    if valid_indices is not None:
        clusters = [valid_indices[c] for c in clusters]

    if return_outliers:
        outliers = np.where(P_unprocessed)[0]
        return clusters, outliers

    return clusters

def project_pointcloud(pc, plane):
    """Project a 3D point cloud onto the defined 2D plane.

    Arguments:
        pc {array} -- Nx3 point cloud array
        plane {array or list} -- Coefficients of plane equation ax + by + cz + d = 0

    Returns:
        array -- Nx3 projected point cloud
    """
    plane_normal = np.array(plane[:3], dtype=np.float32).reshape(3, 1)

    projection_distance = (pc.dot(plane_normal) + plane[-1]) / np.linalg.norm(plane_normal)
    proj_v = projection_distance.dot(plane_normal.reshape(1, 3))
    projection = pc - proj_v

    return projection

def point2plane_distance(point, plane, keepdims=False):
    point, plane = np.array(point).reshape(-1, 3), np.array(plane).reshape(4, 1)
    point = np.hstack((point, [[1]]*len(point)))

    d = point.dot(plane) / np.linalg.norm(plane[:3])

    if not keepdims:
        d = d.reshape(-1)
        if len(d) == 1:
            d = d[0]

    return d
