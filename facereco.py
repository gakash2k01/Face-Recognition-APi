import face_recognition
import pickle
import cv2
import base64
import json
import os
import io
from flask import Flask,request,jsonify
import numpy as np
from PIL import Image

app=Flask(__name__)

@app.route('/RandiRona')
def home():
    return "Hello Worls"

@app.route('/predict',methods=['POST'])
def predict():
    file = request.form.get('image')
    image = base64.b64decode(str(file))
    image = Image.open(io.BytesIO(image))
    image= np.array(image)
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    data = pickle.loads(open('face_enc (1)', "rb").read())
    data1 = pickle.loads(open('usernpb', "rb").read())

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)
            """for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)"""

    fin_name=max(set(names),key=names.count)
    index=data1["name"].index(fin_name)
    place=data1["place"][index]
    branch=data1["branch"][index]
    year=data1['year'][index]
    studying=data1['studying'][index]
    imgs=list(os.listdir('Images/'+fin_name))
    img=cv2.imread('Images/'+fin_name+'/'+imgs[0])
    _, im_arr = cv2.imencode('.jpeg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    result={'name':fin_name,'place':place,'branch':branch,'image':str(im_b64),'year':year,'studying':studying}

    return jsonify(result)




@app.route('/update',methods=['POST'])
def update():
    file = request.form.get('image')
    image = base64.b64decode(str(file))
    image = Image.open(io.BytesIO(image))
    image= np.array(image)
    print('Shape of image is = ',image.shape)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    data = pickle.loads(open('face_enc (1)', "rb").read())
    dict_npb=request.form.get('details')
    print('Type of dictionary = ',type(dict_npb))
    dict_npb = json.loads(dict_npb)
    print("Converted type = ",type(dict_npb))
    print('Extracted dictionry = ',dict_npb)

    for encoding in encodings:
        data['encodings'].append(encoding)
        data['names'].append(dict_npb['name'])

    is_present=0

    try:
        os.mkdir('Images/'+dict_npb['name'])
        is_present=1
    except:
        print('Allready Exists')
        
    cv2.imwrite('Images/'+dict_npb['name']+'/'+str(np.random.randint(1,10000000))+'.jpg',image)
    data1 = pickle.loads(open('usernpb', "rb").read())
    if not(is_present):
        data1['name'].append(dict_npb['name'])
        data1['place'].append(dict_npb['place'])
        data1['branch'].append(dict_npb['branch'])
        data1['year'].append(dict_npb['year'])
        data1['studying'].append(dict_npb['studying'])
    else:
        ind=data1['name'].index(dict_npb['name'])
        data1['place'][ind]=dict_npb['place']
        data1['branch'][ind]=dict_npb['branch']
        data1['year'][ind]=dict_npb['year']
        data1['studying'][ind]=dict_npb['studying']

    fi = open("usernpb", "wb")
    fi.write(pickle.dumps(data1))
    fi.close()
            
    fi = open("face_enc (1)", "wb")
    fi.write(pickle.dumps(data))
    fi.close()
    print('Updated data')
    return {'Hogaya':'OK'}


@app.route('/reset',methods=['GET'])
def reset():
    dat={'encodings':[],'names':[]}

    for fol in list(os.listdir('Images/')):
        pat='Images/'+fol+'/'
        for image in list(os.listdir(pat)):
            img_pat=pat+image
            print(img_pat)
            img=cv2.imread(img_pat)
            print(img.shape)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb,model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                dat['encodings'].append(encoding)
                dat['names'].append(fol)
                    
    fi = open("face_enc (1)", "wb")
    fi.write(pickle.dumps(dat))
    fi.close()
    return {'Hogaya':'OK'}



        


if __name__=='__main__':
    app.run(debug=True)

