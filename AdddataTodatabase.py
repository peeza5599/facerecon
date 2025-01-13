import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://face-recognition-459a6-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref=db.reference('room')
data = {
"987654":
        {
            "name": "Visarut Jaibun",
            "Room_Number": "456",
            "starting_year": 2017,
            "total_attendance": 0,
            "standing": "G",
            "last_attendance_time": "2022-12-11 00:54:34"
        },
"963852":
        {
            "name": "Elonmar",
            "Room_Number": "789",
            "starting_year": 2017,
            "total_attendance": 0,
            "standing": "G",
            "last_attendance_time": "2022-12-11 00:54:34"
        },
"26542":
        {
            "name": "Peerapat",
            "Room_Number": "987",
            "starting_year": 2017,
            "total_attendance": 0,
            "standing": "G",
            "last_attendance_time": "2022-12-11 00:54:34"

        }



}
for key, value in data.items():
    ref.child(key).set(value)