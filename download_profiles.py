import os
import firebase_admin
from firebase_admin import credentials, storage

# ✅ ตั้งค่า Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "face-recognition-459a6.appspot.com"
})

bucket = storage.bucket()
local_profile_folder = "profile"

# ✅ สร้างโฟลเดอร์โปรไฟล์ถ้ายังไม่มี
if not os.path.exists(local_profile_folder):
    os.makedirs(local_profile_folder)

# ✅ ดึงรายชื่อไฟล์ใน Firebase Storage โฟลเดอร์ Images/
blobs = bucket.list_blobs(prefix="Images/")

for blob in blobs:
    filename = blob.name.split("/")[-1]
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # ข้ามไฟล์ที่ไม่ใช่ภาพ

    local_path = os.path.join(local_profile_folder, filename)
    blob.download_to_filename(local_path)
    print(f"📥 ดาวน์โหลด {filename} มาเก็บไว้ที่ {local_path}")

print("✅ ดาวน์โหลดรูปโปรไฟล์ทั้งหมดเรียบร้อย!")
