# myproject/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.http import StreamingHttpResponse, JsonResponse
from django.utils import timezone
import cv2
from myapp.mtcnn_logic import recognizer
from myapp.models import Employee, EmployeeLog  # Import EmployeeLog

def homepage(request):
    return render(request, "home.html")

def adminpage(request):
    return render(request, "admins.html")

def logout_view(request):
    logout(request)
    return redirect('home')

def add_employee(request):
    return render(request, "addemployee.html", {"image_range": range(1, 11)})

def home(request):
    return render(request, 'home.html')

def admin_login(request):
    return render(request, 'adminlogin.html')

# Global variable to store the latest recognized employee
latest_employee = None

def video_feed(request):
    """Stream video feed with face recognition."""
    def generate_frames():
        global latest_employee
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Failed to open camera")
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not accessible\r\n'
            return
        print("Camera opened successfully")

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            employee_data = recognizer.recognize_face(frame)
            if employee_data is not None and 'bounding_box' in employee_data:
                latest_employee = employee_data
                x1 = employee_data['bounding_box']['x1']
                y1 = employee_data['bounding_box']['y1']
                x2 = employee_data['bounding_box']['x2']
                y2 = employee_data['bounding_box']['y2']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{employee_data['name']} ({employee_data['confidence']:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

        cap.release()
        latest_employee = None
        print("Camera released")

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_employee_data(request):
    """Return the latest recognized employee data as JSON."""
    global latest_employee
    if latest_employee:
        return JsonResponse({
            'name': latest_employee['name'],
            'department': latest_employee['department'],
            'employee_id': latest_employee['employee_id'],
            'confidence': latest_employee['confidence']
        })
    return JsonResponse({'status': 'no_employee'})

def log_attendance(request):
    """Handle the 'Proceed' button to log attendance."""
    global latest_employee
    if request.method == "POST" and latest_employee:
        employee_id = latest_employee['employee_id']
        attendance_type = request.POST.get('attendance_type')  # "Time In" or "Time Out"
        
        try:
            employee = Employee.objects.get(employee_id=employee_id)
            today = timezone.now().date()
            
            # Check if there's an existing log for today
            log, created = EmployeeLog.objects.get_or_create(
                employee=employee,
                date=today,
                defaults={'time_in': None, 'time_out': None}
            )
            
            # Update time_in or time_out based on attendance type
            if attendance_type == "Time In" and not log.time_in:
                log.time_in = timezone.now()
            elif attendance_type == "Time Out" and not log.time_out:
                log.time_out = timezone.now()
            
            log.save()
            return JsonResponse({'status': 'success', 'message': f'{attendance_type} logged successfully!'})
        except Employee.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Employee not found'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'No employee data available'})