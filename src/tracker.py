class Tracker():
    def __init__(self, window_width, window_height, margin, ym=1, xm=1, smooth_factor=15):
        self.recent_centers = []
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.ym_per_px = ym # meters per pixel
        self.xm_per_px = xm
        self.smooth_factor = smooth_factor

    def find_window_centroids(self, warped):
        window_centroids = []
        window = np.ones(self.window_width)
        h,w = warped.shape

        l_sum = np.sum(warped[3*h//4:,:w//2], axis=0) # squash bottom of pic into a 1D array
        l_center = np.argmax(np.convolve(window, l_sum)) - self.window_width/2 # find largest area using convolution
        r_sum = np.sum(warped[3*h//4:,w//2:], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width/2 + w//2

        window_centroids.append((l_center, r_center))

        for level in range(1, h//self.window_height):

            image_layer = np.sum(warped[h-(level+1)*self.window_height:h-level*self.window_height,:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin, 0))
            l_max_index = int(min(l_center+offset+self.margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

            r_min_index = int(max(r_center+offset-self.margin, 0))
            r_max_index = int(min(r_center+offset+self.margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)
#         pdb.set_trace()
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

