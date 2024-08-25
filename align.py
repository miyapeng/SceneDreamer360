pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) # 3, 1
new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2
vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2

compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)

compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T

compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
compensate_depth_zero = np.zeros(4)
compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4

pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2

# Calculate for masked pixels
masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

 # Convert image2 to grayscale to create a mask of non-background pixels
gray_image2 = np.mean(image2, axis=-1)  # Assuming image2 is RGB

# Create a mask for non-background pixels (assuming background is black, value of 0)
non_background_mask = gray_image2 > 0

# Project new_pts_coord_world2 to image space
new_pixel_coord_cam2 = np.matmul(K, R.dot(new_pts_coord_world2) + T)
new_pixel_coord_cam2 = new_pixel_coord_cam2[:2] / new_pixel_coord_cam2[2:]

# Round the coordinates to get pixel indices
new_pixel_indices = np.round(new_pixel_coord_cam2).astype(int)

# Ensure the indices are within the image bounds
new_pixel_indices[0] = np.clip(new_pixel_indices[0], 0, W-1)
new_pixel_indices[1] = np.clip(new_pixel_indices[1], 0, H-1)

# Create a mask to exclude new points that fall on non-background pixels in image2
exclude_mask = non_background_mask[new_pixel_indices[1], new_pixel_indices[0]]

# Filter out these points from new_pts_coord_world2
filtered_new_pts_coord_world2 = new_pts_coord_world2[:, ~exclude_mask]
filtered_new_pts_colors2 = new_pts_colors2[~exclude_mask]

# Merge the remaining new points with the existing pts_coord_world
pts_coord_world = np.concatenate((pts_coord_world, filtered_new_pts_coord_world2), axis=-1)
pts_colors = np.concatenate((pts_colors, filtered_new_pts_colors2), axis=0)