from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import numpy as np
import plotly.graph_objects as go
from ultralytics import YOLO
import face_recognition
import datetime
import os
from bson import ObjectId
from pymongo import MongoClient
triangles=[]
import threading
def generate_filename():
    now = datetime.datetime.now()
    st=now.strftime("%Y-%m-%d_%H-%M-%S")
    st="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches/"+st+".jpg"
    return st
def generate_filename1():
    now = datetime.datetime.now()
    st=now.strftime("%Y-%m-%d_%H-%M-%S")
    st="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches_without_halmet/"+st+".jpg"
    return st



            
def timer_callback():
    print("Timer expired!")
    print("changed to False")
    # time.sleep(5)
    collection=db["emergency"]
    filter={'_id':ObjectId('6507ea55c8f670d8cdf91945')}
    document=collection.find_one(filter)
    document['emer']=False
    resul=collection.update_one(filter,{'$set':document})

def set_timer(seconds):
    timer = threading.Timer(seconds, timer_callback)
    # print(f"Timer started for {seconds} seconds.")
    timer.start()


def timer_callback1():
    print("Timer expired!")
    print("changed to False")
    # time.sleep(5)
    collection=db["breach"]
    filter={'_id':ObjectId('6507b9aee35c4230aad8c430')}
    document=collection.find_one(filter)
    document['breach']=False
    resul=collection.update_one(filter,{'$set':document})
def set_timer1(seconds):
    timer = threading.Timer(seconds, timer_callback1)
    timer.start()






# clients = set()
from PIL import Image
app = Flask(__name__,static_folder="static")
app.config['MONGO_URI'] = 'mongodb://localhost:27017/visionsafetyhub'
mongo = MongoClient(app.config['MONGO_URI'])
app.secret_key = 'your_secret_key'
# Load known faces and their names
known_faces = []
known_names = []
# UPLOAD_FOLDER = 'static/uploaded_photos'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Directory containing images of known individuals
known_faces_dir = "C:/Users/uditpc/Pictures/deep_learning/project/static/known_faces"

db = mongo.get_database()
# Load known faces
def update_face_data():
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encoding = face_recognition.face_encodings(image)
            if len(encoding) > 0:
                encoding = encoding[0]  # Assuming one face per image
            else:
    # Handle the case where no faces were detected
                print("No faces were detected in the image of "+filename[:-4]) # Assuming one face per image
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

update_face_data()

    
def generate_frames2():
    
    cap1 = cv2.VideoCapture(0)  
    cap2 = cv2.VideoCapture(1)  
    collection=db["emergency"]
    filter={'_id':ObjectId('6507ea55c8f670d8cdf91945')}
    document=collection.find_one(filter)
    document['emer']=False
    resul=collection.update_one(filter,{'$set':document})
    model1 = YOLO('weights_emer/best.pt')
    while True:
        document=collection.find_one(filter)
        ch=document['emer']
        success1,frame1 = cap1.read()
        success2,frame2 = cap2.read()
        if not success1:
            break
        if not success2:
            break
        
        frame_cp1=frame1.copy()
        frame_cp2=frame2.copy()
        
        h,w,c=frame_cp1.shape
        frame_cp2=cv2.resize(frame_cp2,(w,h))
        frame=cv2.hconcat([frame_cp1,frame_cp2])
        
        results = model1.predict(frame,conf=0.75)
        print(len(results[0]))
        if len(results[0])>0 and ch == False :
            document['emer']=True
            resul=collection.update_one(filter,{'$set':document})
            set_timer(10)
            print("detected and timer started")
        elif len(results[0])>0:
            print("only detected")
        else:
            print("nothing detected")
        frame = results[0].plot()
        
        
        
    
        _, buffer = cv2.imencode('.jpg', frame)#chewck this
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
  
        key=cv2.waitKey(200) 
        if key == ord("q"):
            break
    
    
def generate_frames3():
    # model3=YOLO('train/weights/best.pt')
    cap1 = cv2.VideoCapture(0)  
    cap2 = cv2.VideoCapture(1)  
    collection1=db['breach']
    filter={'_id':ObjectId('6507b9aee35c4230aad8c430')}
    document_id1=ObjectId('6507b9aee35c4230aad8c430')
    document1=collection1.find_one(filter)
    document1['breach']=False
    resul=collection1.update_one(filter,{'$set':document1})
    
    model3=YOLO('C:/Users/uditpc/Pictures/deep_learning/project/train/weights/best.pt')
    while True:
        document1=collection1.find_one(filter)
        ch=document1['breach']
        success1,frame1 = cap1.read()
        success2,frame2 = cap2.read()
        if not success1:
            break
        if not success2:
            break
        
        frame_cp1=frame1.copy()
        frame_cp2=frame2.copy()
        
        h,w,c=frame_cp1.shape
        frame_cp2=cv2.resize(frame_cp2,(w,h))
        frame=cv2.hconcat([frame_cp1,frame_cp2])
        
        results = model3.predict(frame,conf=0.75)
        print(len(results[0]))
        if document1['breach']==False:
            for res in results[0]:
                if res.boxes.cls[0].item()==2 :
                    document1['breach']=True
                    resul=collection1.update_one(filter,{'$set':document1})
                    filename = generate_filename1()
                    cv2.imwrite(filename, frame)
                    set_timer(10)
                    break
        
        
        frame1=results[0].plot()
    
        _, buffer = cv2.imencode('.jpg', frame1)#chewck this
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
  
        key=cv2.waitKey(200) 
        if key == ord("q"):
            break
    



def generate_frames1():
    
    collection = db['triangle']
    document_id=ObjectId('650578be186e80cc35cc0fa6')
    collection1=db['breach']
    filter={'_id':ObjectId('6507b9aee35c4230aad8c430')}
    document_id1=ObjectId('6507b9aee35c4230aad8c430')
    document1=collection1.find_one({'_id':document_id1})
    document = collection.find_one({'_id': document_id})
    document1['breach']=False
    resul=collection1.update_one(filter,{'$set':document1})
    triangles=document['coor']
    cap = cv2.VideoCapture(1)  
    model2=YOLO('yolov8s.pt')
    while True:
        success,frame1 = cap.read()
        if not success:
            break
        if success:
            document1=collection1.find_one({'_id':document_id1})
            frame_cp=frame1.copy()
            # 6507b9aee35c4230aad8c430
            
            
            for triangle_coords in triangles:
                
                pts = np.array(triangle_coords, dtype=np.int32)
                cv2.fillPoly(frame_cp, [pts], (0,0,0))
            results=model2.predict(frame_cp)
            if document1['breach']==False:
                for res in range(0,len(results[0])):
                    if results[0][res].boxes.cls[0].item()==0 :
                        document1['breach']=True
                        resul=collection1.update_one(filter,{'$set':document1})
                        filename = generate_filename()
                        cv2.imwrite(filename, frame1)
                        set_timer1(10)
                        break
                    else:
                        results[0]=np.delete(results[0],res)
                        res=res-1
                    
                        
                    
            absolu=results[0].plot()
            
            _, buffer = cv2.imencode('.jpg', absolu)#chewck this
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   

            key=cv2.waitKey(200) 
            if key == ord("q"):
                break
        else:
            break


        
def generate_frames():
    # cv2.imread("static/images/1.png")
   
    # camera1 = cv2.VideoCapture(1)  # 0 for the default camera (webcam)
    cap = cv2.VideoCapture(0)  # 0 for the default camera (webcam)
    unknown_image=0
    
    collection = db['workers']
    
        
    model = YOLO('weights/best.pt')
    jacket=False
    halmet=False
    while True:
        # success1, frame1 = camera1.read()
        print("inner loop is running xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        success,frame1 = cap.read()
        
        if success:
            
            # Run YOLOv8 inference on the frame
            results = model.predict(frame1,conf=0.80)
            for res in results[0]:
                if res.boxes.cls[0].item()==1.0:
                    jacket=True;
                if res.boxes.cls[0].item()==0.0:
                    halmet=True;
            if jacket == True and halmet == True:
                unknown_image=frame1
                print("accepted")
                
    # Find all face locations and face encodings in the unknown image
                face_locations = face_recognition.face_locations(unknown_image)
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                # Loop through each face found in the unknown image
                # time.sleep(1)
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare the unknown face with known faces
                    matches = face_recognition.compare_faces(known_faces, face_encoding)
                    name = "Unknown"  # Default name if no match is found
                    # If a match is found, use the name of the known face
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        # workers_name[name]=True;
                        document_id=ObjectId('650691168ccb8baa0ab3a5d4')
                        filter={'_id':document_id}
                        document=collection.find_one(filter)
                        document["attend_sheet"][name]=True
                        print("updated "+str(document["attend_sheet"]))
                        result=collection.update_one(filter,{'$set': document}) 
                        print(f"Found {name} in the image at {face_location}")
                for(top, right, bottom, left) in face_locations:
                    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255,128), 2)
                    cv2.rectangle(unknown_image, (0, 0), (1000, 80), ( 168, 191,54,128), -1)
                    cv2.putText(unknown_image,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(238, 222, 6),2)
                    if name !="Unknown":
                        cv2.putText(unknown_image,"YOU ARE ACCEPTED:   "+name.upper(),(100,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),3)
                    _, buffer = cv2.imencode('.jpg', unknown_image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            # time.sleep(2)
                jacket=False
                halmet=False
            annotated_frame = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # for i in range(frame_skip_interval - 1):
            #     cap.read()
            
            # Break the loop if 'q' is pressed
            key=cv2.waitKey(100) 
            jacket=False
            halmet=False
            if key == ord("q"):
                
                break
        else:
            # Break the loop if the end of the video is 
            
            break
       
        
        

        













@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attend')
def attend():
    return render_template('camera.html')
@app.route('/gesture')
def gesture():
    return render_template('camera3.html')
@app.route('/log')
def attend_log():
    return render_template('log.html')
@app.route('/restrict')
def restrict():
    return render_template('image.html')
@app.route('/restrict/camera2')
def restrict_camera():
    return render_template('camera2.html')
@app.route('/about')
def abouta():
    return render_template('about.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/camera4')
def camera4():
    return render_template('camera4.html')
@app.route('/card')
def card():
    return render_template('add_face.html')
@app.route('/delete',methods=['POST'])
def delete():
    text=request.form['text']
    collection=db['workers']
    result = collection.update_one(
            {'_id': ObjectId("650691168ccb8baa0ab3a5d4")},
            {'$unset': {f'attend_sheet.{text}': 1}}
        )
    print(text)
    file_path = 'C:/Users/uditpc/Pictures/deep_learning/project/static/known_faces/'

    try:
    # Attempt to delete the file
        os.remove(file_path+text+".jpg")
        print(f"{file_path} has been deleted.")
        update_face_data()
    except FileNotFoundError:
        print(f"{file_path} not found. File not deleted.")
    except Exception as e:
        print(f"An error occurred while deleting {file_path}: {e}")
    return jsonify({'message': 'successfully'})
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Get the uploaded image file and text value from the request
        image_file = request.files['image']
        text_value = request.form['text']
        
        # Check if the image file is present
        
            # Save the image to a folder (you may need to create the folder if it doesn't exist)
        image_filename = os.path.join(known_faces_dir, text_value+".jpg")
        image_file.save(image_filename)
        collection=db['workers']
        print("updated recently ")
        document=collection.find_one({'_id':ObjectId('650691168ccb8baa0ab3a5d4')})
        print(str(document))
        document['attend_sheet'][text_value]=False
        print(str(document))
            # Process the text value as needed
            # ...
        
        collection.update_one({'_id':ObjectId('650691168ccb8baa0ab3a5d4')},{'$set': document})
        print(str(document))
        update_face_data()
        return jsonify({'message': 'Image and text uploaded successfully'})
       

    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/restrict', methods=['POST'])
def process_triangles():
    try:
        # Receive the JSON data from the frontend
        data = request.get_json()

        # Process the received data (in this example, we simply print it)
        print('Received data from frontend:')
        # for triangle in data:
        #     print(f"Triangle: ({triangle['point1_x']}, {triangle['point1_y']}), ({triangle['point2_x']}, {triangle['point2_y']}), ({triangle['point3_x']}, {triangle['point3_y']}), Side: {triangle['side']}")
        
        # You can perform further processing or database operations here
        db = mongo.get_database()
        collection = db['triangle']
        
        document={
            'name':"tria_coor",
            "coor":data
        }
        # collection.insert_one(document)
        document_id=ObjectId('650578be186e80cc35cc0fa6')
        filter={'_id':document_id}
        
        result=collection.update_one(filter,{'$set': document})
        # Return a response (you can customize this response)
        # response = {'message': 'Data received and processed successfully'}
        return render_template('index.html')

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'message': 'An error occurred'}), 500

@app.route('/data_breach')
def data_breach():
    l=[]
    pathh="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches"
    for i in os.listdir(pathh):
        if i.endswith(".jpg"):
            l.append(i)
            print(i)

    response = app.response_class(
        response=json.dumps(l),
        status=200,
        mimetype='application/json'
    )
    return response
# breaches_without_halmet
@app.route('/data_breach1')
def data_breach1():
    l=[]
    pathh="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches_without_halmet"
    for i in os.listdir(pathh):
        if i.endswith(".jpg"):
            l.append(i)
            print(i)

    response = app.response_class(
        response=json.dumps(l),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/data')
def send_data():
    collection = db['workers']
    document_id=ObjectId('650691168ccb8baa0ab3a5d4')
    filter={'_id':document_id}
    document=collection.find_one(filter)
    print(document)
    response = app.response_class(
        response=json.dumps(document["attend_sheet"]),
        status=200,
        mimetype='application/json'
    )
    return response
@app.route('/your-backend-endpoint')
def back_data():
    collection=db['breach']
    document_id=ObjectId('6507b9aee35c4230aad8c430')
    filter={'_id':document_id}
    document=collection.find_one(filter)
    print(str(document))
    response=app.response_class(
        response=json.dumps(document["breach"]),
        status=200,
        mimetype='application/json'
    )
    return response
@app.route('/del_breach_data')
def del_breach():
    pathh="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches"
    for i in os.listdir(pathh):
        file_path = os.path.join(pathh, i)
        os.remove(file_path)
    response=app.response_class(
        response=json.dumps(True),
        status=200,
        mimetype='application/json'
    )
    return response
@app.route('/del_breach_data1')
def del_breach1():
    pathh="C:/Users/uditpc/Pictures/deep_learning/project/static/breaches_without_halmet"
    for i in os.listdir(pathh):
        file_path = os.path.join(pathh, i)
        os.remove(file_path)
    response=app.response_class(
        response=json.dumps(True),
        status=200,
        mimetype='application/json'
    )
    return response
@app.route('/restrict/camera2/breach')
def calculate_graph():
    days=[]
    times=[]
    for filename in os.listdir("C:/Users/uditpc/Pictures/deep_learning/project/static/breaches"):
        if filename.endswith(".jpg"):
            fname=filename[:4]+filename[5:7]+filename[8:10]
            if len(days)==0:

                
                days.append(int(fname))
                times.append(1)
            else :
                if days[len(days)-1]==int(fname):
                    times[len(times)-1]=times[len(times)-1]+1
                else:
                    days.append(int(fname))
                    times.append(1)
    
# Sample data
    x_data =days
    y_data=times
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a bar trace
    fig.add_trace(go.Bar(x=x_data, y=y_data, name='Bar Chart'))
    
    # Add a line trace
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name='Line Chart'))
    
    # Update layout and labels
    fig.update_layout(title='breaches vs date GRAPH', xaxis_title='date(YY.YYMMDD)', yaxis_title='no of breaches')
    
    # Show the plot
    chart_file = 'static/chart.html'
    fig.write_html(chart_file)

    # Render the HTML template
    return render_template('breach.html', chart_file=chart_file)
@app.route('/restrict/camera2/breach2')
def calculate_graph2():
    days=[]
    times=[]
    for filename in os.listdir("C:/Users/uditpc/Pictures/deep_learning/project/static/breaches_without_halmet"):
        if filename.endswith(".jpg"):
            fname=filename[:4]+filename[5:7]+filename[8:10]
            if len(days)==0:

                
                days.append(int(fname))
                times.append(1)
            else :
                if days[len(days)-1]==int(fname):
                    times[len(times)-1]=times[len(times)-1]+1
                else:
                    days.append(int(fname))
                    times.append(1)
    
# Sample data
    x_data =days
    y_data=times
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a bar trace
    fig.add_trace(go.Bar(x=x_data, y=y_data, name='Bar Chart'))
    
    # Add a line trace
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name='Line Chart'))
    
    # Update layout and labels
    fig.update_layout(title='breaches vs date GRAPH', xaxis_title='date(YY.YYMMDD)', yaxis_title='no of breaches')
    
    # Show the plot
    chart_file = 'static/chart.html'
    fig.write_html(chart_file)

    # Render the HTML template
    return render_template('breach1.html', chart_file=chart_file)
    
   
@app.route('/your-backend-endpoint_camera3')
def back_data_camera3():
    collection=db['emergency']
    document_id=ObjectId('6507ea55c8f670d8cdf91945')
    filter={'_id':document_id}
    document=collection.find_one(filter)
    print(str(document))
    response=app.response_class(
        response=json.dumps(document["emer"]),
        status=200,
        mimetype='application/json'
    )
    return response
@app.route('/video_feed')
def video_feed():
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed3')
def video_feed3():
    return Response(generate_frames3(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    

