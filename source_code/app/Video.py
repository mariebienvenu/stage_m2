import cv2
import numpy as np

class Video:
   
    def __init__(self, filepath, verbose=0):
        self.filepath = filepath
                
        vid_capture = cv2.VideoCapture(filepath)
        assert vid_capture.isOpened(), "Error opening the video file"
        ret, frame = vid_capture.read()
        assert ret, "Error reading the video file"

        self.fps = vid_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height, self.frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.wait_time = int(1000/self.fps) #in milliseconds
        self.n_channels = frame.shape[2] #cv2.CAP_PROP_CHANNEL and cv2.CAP_PROP_VIDEO_TOTAL_CHANNELS do not work as intended
        self.is_loaded = False

        vid_capture.release()

        if verbose>0:
            print(f'VIDEO DETAILS: \n\t Frame rate: {self.fps} FPS \n\t Frame count: {self.frame_count} \n\t Frame dimension: ({self.frame_height},{self.frame_width}) \n\t Number of color channels: {self.n_channels}')

    def play(self, close_key='q'):
        vid_capture = cv2.VideoCapture(self.filepath)
        while(vid_capture.isOpened()):
            ret, frame = vid_capture.read() #bool, numpy array of frame
            if ret == True:
                cv2.imshow('Video',frame)
                key = cv2.waitKey(self.wait_time)
                if key == ord(close_key):
                    break
            else:
                break
        vid_capture.release()
        cv2.destroyAllWindows()

    def load(self, verbose=0):
        if self.is_loaded:
            if verbose>0:
                print('Video already loaded in memory.')
            return self.video_content
        self.video_content = np.zeros((self.frame_count, self.frame_height, self.frame_width, self.n_channels), dtype=np.uint8)
        vid_capture = cv2.VideoCapture(self.filepath)
        index = 0
        while(vid_capture.isOpened()):
            ret, frame = vid_capture.read() #bool, numpy array of frame
            if ret == True:
                self.video_content[index, :, :, :] = np.copy(frame)
            else:
                break
            index += 1
        vid_capture.release()
        cv2.destroyAllWindows()
        self.is_loaded = True
        if verbose>0:
            print('Video now loaded in memory.')
        return self.video_content
    
    def play_frame_by_frame(self, close_key='q', next_key='z', previous_key='a', start_frame=None):
        self.load()
        start = start_frame if start_frame is not None else self.frame_count//2
        diff = 0
        while True:
            current = max(0, min(start+diff, self.frame_count))
            cv2.imshow("Frame by frame", self.video_content[current,:,:,:])
            key = cv2.waitKey(0)
            if key == ord(close_key):
                cv2.destroyWindow("Frame by frame")
                break
            elif key == ord(previous_key) and start+diff>=1:
                diff -= 1
            elif key == ord(next_key) and start+diff<self.frame_count-1:
                diff += 1
            print(f'Displaying frame {start+diff}')

    def get_frame(self, index):
        self.load()
        assert index>=0 and index<self.frame_count, f"Index {index} out of video range [0, {self.frame_count}]."
        return np.copy(self.video_content[index,:,:,:])
    
    @staticmethod
    def from_array(array, filepath='/tmp.mp4', fps=30, verbose=0):
        # saves, returns Video object
        frame_count = array.shape[0]
        frame_height, frame_width = array.shape[1], array.shape[2]
        if len(array.shape)<4:
            array = np.expand_dims(array)
        n_channels = array.shape[3]
        is_color = n_channels != 1
        fourcc = 'mp4v' if 'mp4' in filepath[len(filepath)-3:] else 'XVID' # MP4 or AVI codec
        vid_writer = cv2.VideoWriter(
            filepath,
            fourcc=cv2.VideoWriter_fourcc(*fourcc),
            fps=fps,
            frameSize=(frame_width, frame_height),
            isColor=is_color
        )
        
        for i in range(frame_count):
            frame = array[i,:,:,:]
            vid_writer.write(frame)
        vid_writer.release()

        return Video(filepath, verbose)
