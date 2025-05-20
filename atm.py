import os
import tkinter as tk
from tkinter import messagebox
import cv2
import csv
import hashlib
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models

# Suppress TensorFlow warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
CSV_FILE = "user_data.csv"
IMAGE_DIR = "faces"
IMAGE_SIZE = (100, 100)  # Face image size for CNN input
PRIMARY_COLOR = "#00A3E0"  # Color that resembles the provided app
SECONDARY_COLOR = "#E2F0F7"  # Light color for backgrounds
TEXT_COLOR = "#333333"  # Dark text color
BUTTON_COLOR = "#00A3E0"  # Button color
BUTTON_HOVER_COLOR = "#00A3E0"  # Darker shade for button hover
FONT_NAME = "Arial"  # Changed to a more common font

os.makedirs(IMAGE_DIR, exist_ok=True)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def load_users():
    users = {}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users[row['name']] = row
    return users

def save_users(users):
    with open(CSV_FILE, 'w', newline='') as f:
        fieldnames = ['name', 'password', 'face_path', 'balance', 'phone']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for user in users.values():
            writer.writerow(user)

def on_enter_button(e):
    e.widget['background'] = BUTTON_HOVER_COLOR

def on_leave_button(e):
    e.widget['background'] = BUTTON_COLOR

def capture_face_image(name):
    capture_window = tk.Toplevel()
    capture_window.title(f"Capture Face - {name}")
    capture_window.geometry("640x480")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera")
        capture_window.destroy()
        return None

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Convert frame to Tkinter compatible format
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            label.imgtk = imgtk
            label.configure(image=imgtk)
        label.after(10, update_frame)

    

def capture_face_image(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera")
        return None
    messagebox.showinfo("Info", "Position your face in front of the camera and press 'Space' to capture.")
    face_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "Press Space to capture, ESC to cancel", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.imshow("Capture Face - "+name, frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
        elif k % 256 == 32:
            face_img = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    if face_img is not None:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected. Please try again.")
            return None
        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        face_crop = face_img[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, IMAGE_SIZE)
        face_path = os.path.join(IMAGE_DIR, f"{name}.jpg")
        cv2.imwrite(face_path, face_resized)
        return face_path
    return None

def preprocess_face_image(face_path):
    img = cv2.imread(face_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float") / 255.0
    img = img.reshape((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    return img

def build_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    return model

def get_face_embedding(model, face_img):
    face_img = np.expand_dims(face_img, axis=0)
    embedding = model.predict(face_img)[0]
    return embedding

def verify_face(model, enrolled_embedding, test_embedding, threshold=6.0):
    dist = np.linalg.norm(enrolled_embedding - test_embedding)
    return dist < threshold

class ATMFaceRecognitionApp:
    def init(self, root):
        self.root = root
        self.root.title("ATM Security - Facial Recognition")
        self.root.config(bg=SECONDARY_COLOR)
        self.root.attributes('-fullscreen', True)
        self.root.resizable(False, False)
        self.users = load_users()
        self.model = build_cnn_model()
        self.enrolled_embeddings = {}
        self.load_all_embeddings()
        self.current_user = None

        self.main_frame = tk.Frame(root, bg=SECONDARY_COLOR)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.title_label = tk.Label(self.main_frame, text="SBI BANK", font=(FONT_NAME, 22, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR)
        self.title_label.pack(pady=(0,25))

        self.btn_enroll = tk.Button(self.main_frame, text="Enroll", width=30, height=2, bg=BUTTON_COLOR, fg='white',
                                    font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                                    command=self.enroll_screen)
        self.btn_enroll.pack(pady=10)
        self.btn_enroll.bind("<Enter>", on_enter_button)
        self.btn_enroll.bind("<Leave>", on_leave_button)

        self.btn_login = tk.Button(self.main_frame, text="Login", width=30, height=2, bg=BUTTON_COLOR, fg='white',
                                   font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                                   command=self.login_screen)
        self.btn_login.pack(pady=10)
        self.btn_login.bind("<Enter>", on_enter_button)
        self.btn_login.bind("<Leave>", on_leave_button)

        self.btn_exit = tk.Button(self.main_frame, text="Exit", width=30, height=2, bg="#e94b3c", fg='white',
                                   font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground="#b8362a",
                                   command=root.quit)
        self.btn_exit.pack(pady=10)
        self.btn_exit.bind("<Enter>", lambda e: e.widget.config(bg="#b8362a"))
        self.btn_exit.bind("<Leave>", lambda e: e.widget.config(bg="#e94b3c"))

    def load_all_embeddings(self):
        for name, user in self.users.items():
            face_path = user['face_path']
            if os.path.exists(face_path):
                face_img = preprocess_face_image(face_path)
                embedding = get_face_embedding(self.model, face_img)
                self.enrolled_embeddings[name] = embedding

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def main_menu(self):
        self.clear_frame()
        self.init(self.root)

    def enroll_screen(self):
        self.clear_frame()
        frame = tk.Frame(self.root, bg=SECONDARY_COLOR, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Enroll New User", font=(FONT_NAME, 20, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR).pack(pady=15)

        tk.Label(frame, text="Full Name:", font=(FONT_NAME, 14), fg=TEXT_COLOR, bg=SECONDARY_COLOR).pack(anchor="w")
        name_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, relief='solid', borderwidth=1)
        name_entry.pack(pady=8)

        tk.Label(frame, text="Password:", font=(FONT_NAME, 14), fg=TEXT_COLOR, bg=SECONDARY_COLOR).pack(anchor="w")
        pwd_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, show="*", relief='solid', borderwidth=1)
        pwd_entry.pack(pady=8)
        
        tk.Label(frame, text="Phone Number:", font=(FONT_NAME, 14), fg=TEXT_COLOR, bg=SECONDARY_COLOR).pack(anchor="w")
        phone_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, relief='solid', borderwidth=1)
        phone_entry.pack(pady=8)

        def enroll_action():
            name = name_entry.get().strip()
            pwd = pwd_entry.get().strip()
            phone = phone_entry.get().strip()
            if not name or not pwd or not phone:
                messagebox.showwarning("Warning", "Please fill all the fields")
                return
            if name in self.users:
                messagebox.showwarning("Warning", "User already exists")
                return
            self.root.withdraw()
            face_path = capture_face_image(name)
            self.root.deiconify()

            if not face_path:
                return
            hashed_pwd = hash_password(pwd)
            user = {'name': name, 'password': hashed_pwd, 'face_path': face_path, 'balance': '0', 'phone': phone}
            self.users[name] = user
            save_users(self.users)
            face_img = preprocess_face_image(face_path)
            embedding = get_face_embedding(self.model, face_img)
            self.enrolled_embeddings[name] = embedding

            messagebox.showinfo("Success", "User enrolled successfully!")
            self.main_menu()

        btn_frame = tk.Frame(frame, bg=SECONDARY_COLOR)
        btn_frame.pack(pady=20)

        btn_enroll = tk.Button(btn_frame, text="Enroll", width=15, height=2, bg=BUTTON_COLOR, fg='white',
                               font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                               command=enroll_action)
        btn_enroll.pack(side="left", padx=10)
        btn_enroll.bind("<Enter>", on_enter_button)
        btn_enroll.bind("<Leave>", on_leave_button)

        btn_back = tk.Button(btn_frame, text="Back", width=15, height=2, bg="#888", fg='white',
                             font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground="#666",
                             command=self.main_menu)
        btn_back.pack(side="left", padx=10)
        btn_back.bind("<Enter>", lambda e: e.widget.config(bg="#666"))
        btn_back.bind("<Leave>", lambda e: e.widget.config(bg="#888"))

    def login_screen(self):
        self.clear_frame()
        frame = tk.Frame(self.root, bg=SECONDARY_COLOR, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="User Login", font=(FONT_NAME, 20, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR).pack(pady=15)

        tk.Label(frame, text="Full Name:", font=(FONT_NAME, 14), fg=TEXT_COLOR, bg=SECONDARY_COLOR).pack(anchor="w")
        name_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, relief='solid', borderwidth=1)
        name_entry.pack(pady=8)

        tk.Label(frame, text="Password:", font=(FONT_NAME, 14), fg=TEXT_COLOR, bg=SECONDARY_COLOR).pack(anchor="w")
        pwd_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, show="*", relief='solid', borderwidth=1)
        pwd_entry.pack(pady=8)

        def login_action():
            name = name_entry.get().strip()
            pwd = pwd_entry.get().strip()
            if not name or not pwd:
                messagebox.showwarning("Warning", "Please enter both name and password")
                return
            if name not in self.users:
                messagebox.showwarning("Warning", "User does not exist")
                return
            hashed_pwd = hash_password(pwd)
            if self.users[name]['password'] != hashed_pwd:
                messagebox.showwarning("Warning", "Incorrect password")
                return
        
            self.root.withdraw()
            face_path = capture_face_image(name + " (Login)")
            self.root.deiconify()
            if not face_path:
                return
            face_img = preprocess_face_image(face_path)
            test_embedding = get_face_embedding(self.model, face_img)
            enrolled_embedding = self.enrolled_embeddings.get(name)

            if enrolled_embedding is None or len(enrolled_embedding) == 0:
                messagebox.showerror("Error", "Enrolled face data missing for user.")
                return

            if verify_face(self.model, enrolled_embedding, test_embedding, threshold=6.0):
                self.current_user = name
                messagebox.showinfo("Success", f"Welcome {name}!")
                self.dashboard_screen()
            else:
                messagebox.showwarning("Warning", "Face verification failed! Access denied.")
        
            try:
                os.remove(face_path)
            except:
                pass


        btn_frame = tk.Frame(frame, bg=SECONDARY_COLOR)
        btn_frame.pack(pady=20)

        btn_login = tk.Button(btn_frame, text="Login", width=15, height=2, bg=BUTTON_COLOR, fg='white',
                              font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                              command=login_action)
        btn_login.pack(side="left", padx=10)
        btn_login.bind("<Enter>", on_enter_button)
        btn_login.bind("<Leave>", on_leave_button)

        btn_back = tk.Button(btn_frame, text="Back", width=15, height=2, bg="#888", fg='white',
                             font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground="#666",
                             command=self.main_menu)
        btn_back.pack(side="left", padx=10)
        btn_back.bind("<Enter>", lambda e: e.widget.config(bg="#666"))
        btn_back.bind("<Leave>", lambda e: e.widget.config(bg="#888"))

    def dashboard_screen(self):
        self.clear_frame()
        frame = tk.Frame(self.root, bg=SECONDARY_COLOR, padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=f"Welcome, {self.current_user}", font=(FONT_NAME, 20, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR).pack(pady=15)
        btns = [
            ("Withdraw", self.withdraw_screen),
            ("Deposit", self.deposit_screen),
            ("Check Balance", self.balance_screen),
            ("Logout", self.logout)
        ]
        for text, cmd in btns:
            btn = tk.Button(frame, text=text, width=30, height=2, bg=BUTTON_COLOR, fg='white',
                            font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                            command=cmd)
            btn.pack(pady=8)
            btn.bind("<Enter>", on_enter_button)
            btn.bind("<Leave>", on_leave_button)

    def withdraw_screen(self):
        self.clear_frame()
        frame = tk.Frame(self.root, bg=SECONDARY_COLOR, padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Withdraw Amount", font=(FONT_NAME, 20, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR).pack(pady=15)
        amount_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, relief='solid', borderwidth=1)
        amount_entry.pack(pady=8)

        def withdraw_action():
            amt_str = amount_entry.get().strip()
            if not amt_str.isdigit():
                messagebox.showwarning("Warning", "Enter a valid amount")
                return
            amt = int(amt_str)
            if amt <= 0:
                messagebox.showwarning("Warning", "Amount must be positive")
                return
            balance = float(self.users[self.current_user]['balance'])
            if amt > balance:
                messagebox.showwarning("Warning", "Insufficient balance")
                return
            balance -= amt
            self.users[self.current_user]['balance'] = str(balance)
            save_users(self.users)
            messagebox.showinfo("Success", f"Withdrawal successful! New balance: {balance}")
            self.dashboard_screen()

        btn_frame = tk.Frame(frame, bg=SECONDARY_COLOR)
        btn_frame.pack(pady=20)

        btn_withdraw = tk.Button(btn_frame, text="Withdraw", width=15, height=2, bg=BUTTON_COLOR, fg='white',
                                 font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                                 command=withdraw_action)
        btn_withdraw.pack(side="left", padx=10)
        btn_withdraw.bind("<Enter>", on_enter_button)
        btn_withdraw.bind("<Leave>", on_leave_button)

        btn_back = tk.Button(btn_frame, text="Back", width=15, height=2, bg="#888", fg='white',
                             font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground="#666",
                             command=self.dashboard_screen)
        btn_back.pack(side="left", padx=10)
        btn_back.bind("<Enter>", lambda e: e.widget.config(bg="#666"))
        btn_back.bind("<Leave>", lambda e: e.widget.config(bg="#888"))

    def deposit_screen(self):
        self.clear_frame()
        frame = tk.Frame(self.root, bg=SECONDARY_COLOR, padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Deposit Amount", font=(FONT_NAME, 20, 'bold'), fg=PRIMARY_COLOR, bg=SECONDARY_COLOR).pack(pady=15)
        amount_entry = tk.Entry(frame, font=(FONT_NAME, 14), width=30, relief='solid', borderwidth=1)
        amount_entry.pack(pady=8)

        def deposit_action():
            amt_str = amount_entry.get().strip()
            if not amt_str.isdigit():
                messagebox.showwarning("Warning", "Enter a valid amount")
                return
            amt = int(amt_str)
            if amt <= 0:
                messagebox.showwarning("Warning", "Amount must be positive")
                return
            balance = float(self.users[self.current_user]['balance'])
            balance += amt
            self.users[self.current_user]['balance'] = str(balance)
            save_users(self.users)
            messagebox.showinfo("Success", f"Deposit successful! New balance: {balance}")
            self.dashboard_screen()

        btn_frame = tk.Frame(frame, bg=SECONDARY_COLOR)
        btn_frame.pack(pady=20)

        btn_deposit = tk.Button(btn_frame, text="Deposit", width=15, height=2, bg=BUTTON_COLOR, fg='white',
                                font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground=BUTTON_HOVER_COLOR,
                                command=deposit_action)
        btn_deposit.pack(side="left", padx=10)
        btn_deposit.bind("<Enter>", on_enter_button)
        btn_deposit.bind("<Leave>", on_leave_button)

        btn_back = tk.Button(btn_frame, text="Back", width=15, height=2, bg="#888", fg='white',
                             font=(FONT_NAME, 14, 'bold'), borderwidth=0, activebackground="#666",
                             command=self.dashboard_screen)
        btn_back.pack(side="left", padx=10)
        btn_back.bind("<Enter>", lambda e: e.widget.config(bg="#666"))
        btn_back.bind("<Leave>", lambda e: e.widget.config(bg="#888"))

    def balance_screen(self):
        balance = float(self.users[self.current_user]['balance'])
        messagebox.showinfo("Balance", f"Your current balance is: {balance}")

    def logout(self):
        self.current_user = None
        messagebox.showinfo("Logout", "You have been logged out.")
        self.main_menu()

if name == "main":
    root = tk.Tk()
    app = ATMFaceRecognitionApp(root)
    root.mainloop()