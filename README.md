# Face Recognition-Based Attendance System


## 📌 Prerequisites
- **Python**: Version 3.8 or higher
- **Git**: To clone the repository
- **Microsoft SQL Server**: For the database (configured in `settings.py`)
- **ODBC Driver 17 for SQL Server**: Required for database connectivity
- **Webcam**: For facial recognition functionality
- **GPU (optional)**: For faster face recognition processing with CUDA support


## 🛠️ Setup Instructions


### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```


### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate


# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```


### 3. Install Dependencies

Install the dependencies:
```bash
pip install django==5.1.6 pyodbc Pillow opencv-python numpy torch torchvision facenet-pytorch

```
**Note:** If using a GPU, ensure you have CUDA installed and install the appropriate PyTorch version with CUDA support. Refer to PyTorch's official site for details.


### 4. Configure the Database
The project uses Microsoft SQL Server with Windows Authentication. Update the `DATABASES` setting in `myproject/settings.py` if needed:
```python
DATABASES = {
    'default': {
        'ENGINE': 'mssql',
        'NAME': 'DTR',  # Your database name
        'USER': '',     # Leave empty for Windows Authentication
        'PASSWORD': '',  # Leave empty for Windows Authentication
        'HOST': 'OWEL',  # Your SQL Server name
        'PORT': '',     # Default is 1433
        'OPTIONS': {
            'driver': 'ODBC Driver 17 for SQL Server',
            'trusted_connection': 'yes',
        },
    }
}
```
Ensure the `DTR` database exists in your SQL Server instance.
Install the [ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/microsoft-odbc-driver-for-sql-server) if not already present.


### 5. Apply Migrations
Run the following commands to set up the database schema:
```bash
python manage.py makemigrations
python manage.py migrate
```


### 6. Create an Admin User
Create a superuser to access the admin panel:
```bash
python manage.py createsuperuser
```
Follow the prompts to set up the username and password.



### 7. Run the Development Server
Start the Django development server:
```bash
.venv\Scripts\Activate

cd myproject    

python manage.py runserver
```
Access the application at `http://127.0.0.1:8000/`.


## 🚀 Usage
- **Homepage:** Click "Time In" or "Time Out" to start the webcam feed for attendance logging.
- **Admin Login:** Go to `/adminlogin/` or click the admin icon to log in.
- **Admin Dashboard:** Manage employees, view logs, and retrain the model at `/admins/`.
- **Add/Edit Employees:** Upload up to 10 images per employee for accurate recognition.
- **Retrain Model:** Use the "Retrain Model" button in the admin dashboard to update the face recognition classifier after adding or editing employees.


## 🔧 Troubleshooting
- **Webcam Issues:** Ensure your webcam is connected and not in use by another application.
- **Database Errors:** Verify SQL Server is running and the ODBC driver is installed.
- **Recognition Failures:** Ensure employee images are clear and well-lit. Retrain the model if recognition accuracy is low.
- **Dependencies:** If errors occur, check package versions or reinstall dependencies.
