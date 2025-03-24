import pickle

with open("EncodeFile.p", "rb") as file:
    data = pickle.load(file)

# แสดงข้อมูลทั้งหมด
print("✅ จำนวน ID ทั้งหมด:", len(data))
for user_id, encoding in data.items():
    print(f"🧑‍💻 ID: {user_id} | ขนาด encoding: {len(encoding)}")
    print(encoding[:], "...")  # แสดงเฉพาะ 5 ค่าแรก
