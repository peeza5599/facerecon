import os
import hashlib
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

# ✅ Helper: คำนวณ Hash ของไฟล์
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ✅ 1. โหลดชื่อไฟล์จาก Firebase
firebase_files = {}
blobs = bucket.list_blobs(prefix="Images/")

for blob in blobs:
    filename = blob.name.split("/")[-1]
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    firebase_files[filename] = blob

# ✅ 2. เปรียบเทียบ และดาวน์โหลดเฉพาะไฟล์ใหม่/ที่เปลี่ยน
for filename, blob in firebase_files.items():
    local_path = os.path.join(local_profile_folder, filename)
    should_download = True

    if os.path.exists(local_path):
        # เปรียบเทียบ hash ว่ารูปเปลี่ยนไหม
        with open(local_path, 'rb') as f:
            local_hash = hashlib.md5(f.read()).hexdigest()
        cloud_hash = hashlib.md5(blob.download_as_bytes()).hexdigest()

        if local_hash == cloud_hash:
            should_download = False  # ไม่ต้องโหลดใหม่

    if should_download:
        blob.download_to_filename(local_path)
        print(f"📥 ดาวน์โหลดใหม่/อัปเดต: {filename}")

# ✅ 3. ลบรูปในเครื่องที่ไม่มีอยู่ใน Firebase
local_files = set(os.listdir(local_profile_folder))
firebase_file_names = set(firebase_files.keys())

files_to_delete = local_files - firebase_file_names

for filename in files_to_delete:
    path_to_delete = os.path.join(local_profile_folder, filename)
    os.remove(path_to_delete)
    print(f"🗑️ ลบไฟล์ที่ไม่มีใน Firebase: {filename}")

print("✅ ซิงค์โปรไฟล์เสร็จเรียบร้อย!")