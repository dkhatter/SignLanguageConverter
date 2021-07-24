import sys
import cv2
import mediapipe as mp
import numpy as np



def detect(frms,blnk,filepath):
    try:
        blnk=int(blnk)
        Total_frames=int(frms)
        max_num_hands = 1
        gesture = {
            0:'Good', 1:'My', 2:'Afternoon', 3:'Morning', 4:'Bad', 5:'Hello',
            6:'Bye', 7:'You', 8:'Yes', 9:'No', 10:'Please', 11:'I'
        }
        rps_gesture = {0: 'Good', 1: 'My', 2: 'Afternoon', 3: 'Morning', 4: 'Bad', 5: 'Hello', 7: 'You',6:'BYE',11:'I'}

        # MediaPipe hands model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8)

        # Gesture recognition model
        file = np.genfromtxt(filepath, delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)

        cap = cv2.VideoCapture(0)
        sent=''
        blankframe=0
        global predict
        predict = []
        while cap.isOpened():

            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is None:
                blankframe=blankframe+1
                print(blankframe)
            else:
                blankframe=0
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    # Inference gesture
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 9)
                    #ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])

                    # Draw gesture result
                    if idx in rps_gesture.keys():
                        predict.append(rps_gesture[idx].upper())
                        print('--------')
                        print(len(predict))
                        print(predict)
                        print(np.unique(predict[-10:]))
                        print('s== ' + sent)
                        print('--------')

                        print('gesture= ' + rps_gesture[idx].upper() )
                        if len(predict)>=Total_frames:
                            cnt=1
                            unique=False
                            while cnt<Total_frames+1:
                                print(cnt)
                                if predict[len(predict)-cnt]==rps_gesture[idx].upper():
                                    unique=True
                                else:
                                    unique=False
                                cnt=cnt+1
                            if unique==True:
                                print('$$$$$$$$$$$$$$')
                                print('s====  ' +  sent)
                                print('$$$$$$$$$$$$$$')
                                sent = sent + rps_gesture[idx].upper() + " "
                                print('******')
                                print('new sent====  ' + sent)


                                del predict[:]
                                print('******')
                        cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    # Other gestures
                    # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            #cv2.putText(img, text=sent, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),thickness=2)
            cv2.rectangle(img, (0, 0), (800, 30), (245, 117, 16), -1)
            cv2.putText(img, sent, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Raw Feed', img)
            if blankframe>blnk:
                del predict[:]
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return sent
    except Exception as e:
        result = "Exception: " + str(e)
        return result
if __name__ == "__main__":
    #detect(sys.argv[1], sys.argv[2],sys.argv[3])
    detect('25', 10, 'C:\\Users\\sidwived.AD-ONE\\Desktop\\HyperInnovate\\Techgig\\Round_2\\Test_1\\data\\gesture_train.csv')
