from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, session
from PIL import Image
import os
from datetime import datetime
import mysql.connector
import cv2
import numpy as np
import numbers
import base64
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',         
    'database': 'image_db'  
}

def insert_image_info_mysql(info):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            INSERT INTO images (
                filename, format, width, height, color_mode, bit_depth, compression,
                dpi_x, dpi_y, physical_width_cm, physical_height_cm,
                avg_r, avg_g, avg_b, size, uploaded_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            info['filename'], info['format'], info['width'], info['height'],
            info['color_mode'], info['bit_depth'], info['compression'],
            info['dpi_x'], info['dpi_y'], info['physical_width_cm'], info['physical_height_cm'],
            info['avg_r'], info['avg_g'], info['avg_b'], info['size'], datetime.now()
        )
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print("MySQL Error:", err)

def home_processed_image(filename):
    try:
        def to_float(val):
            try:
                if isinstance(val, numbers.Number):
                    return float(val)
                return float(val.numerator) / float(val.denominator)
            except Exception:
                return 72.0 

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            print(f"[ERROR] Image file not found: {filepath}")
            return

        img = Image.open(filepath)
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            print(f"[ERROR] cv2 could not read image: {filepath}")
            return

        avg_color = img_cv.mean(axis=0).mean(axis=0)
        dpi_raw = img.info.get("dpi", (72, 72))
        dpi_x = to_float(dpi_raw[0])
        dpi_y = to_float(dpi_raw[1])

        info = {
            'filename': filename,
            'format': img.format,
            'width': img.width,
            'height': img.height,
            'color_mode': img.mode,
            'bit_depth': len(img.getbands()) * 8,
            'compression': img.info.get("compression", "N/A"),
            'dpi_x': round(dpi_x, 2),
            'dpi_y': round(dpi_y, 2),
            'physical_width_cm': round((img.width / dpi_x) * 2.54, 2),
            'physical_height_cm': round((img.height / dpi_y) * 2.54, 2),
            'avg_r': round(avg_color[2], 2),
            'avg_g': round(avg_color[1], 2),
            'avg_b': round(avg_color[0], 2),
            'uploaded_at': datetime.now().strftime("%d %B %Y, %I:%M %p"),
            'image_path': filename,
            'size': os.path.getsize(filepath)
        }

        print(f"[INFO] Inserting image info to DB for: {filename}")
        insert_image_info_mysql(info)

    except Exception as e:
        print(f"[EXCEPTION] Failed to process image: {filename} | Error: {str(e)}")

def insert_object_measurements(image_filename, object_name, height_cm, width_cm):
    try:
        if not object_name or len(object_name.strip()) == 0:
            raise ValueError("Object name cannot be empty")
        if len(object_name) > 50:
            raise ValueError("Object name cannot exceed 50 characters")

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            INSERT INTO object_measurements (image_filename, object_name, height_cm, width_cm, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (image_filename, object_name.strip(), float(height_cm), float(width_cm), datetime.now())
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        raise
    except ValueError as ve:
        print("Validation Error:", ve)
        raise

PIXELS_PER_CM = 37.795
A4_WIDTH_CM = 21
A4_HEIGHT_CM = 29.7

def detect_objects_from_camera(image_data, filename):
    """Detect objects from camera captured image"""
    try:
        # Convert base64 to OpenCV image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_cv is None:
            return None, []

        # Resize for consistent processing
        image_cv = cv2.resize(image_cv, (1200, 1600))
        paper_pts = np.array([[0, 0], [1200, 0], [0, 1600], [1200, 1600]], dtype="float32")
        output_size = (int(A4_WIDTH_CM * PIXELS_PER_CM), int(A4_HEIGHT_CM * PIXELS_PER_CM))
        dst_pts = np.array([[0, 0],
                            [output_size[0] - 1, 0],
                            [0, output_size[1] - 1],
                            [output_size[0] - 1, output_size[1] - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(paper_pts, dst_pts)
        warped = cv2.warpPerspective(image_cv, matrix, output_size)

        # Object detection
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                box = cv2.boxPoints(rect)
                box = box.astype(int)

                if np.all(box[:, 0] >= 5) and np.all(box[:, 0] <= output_size[0] - 5) and \
                   np.all(box[:, 1] >= 5) and np.all(box[:, 1] <= output_size[1] - 5):

                    width_cm = w / PIXELS_PER_CM
                    height_cm = h / PIXELS_PER_CM
                    label = f"{width_cm:.1f}x{height_cm:.1f} cm"

                    # Draw contours and labels
                    cv2.drawContours(warped, [box], 0, (0, 255, 0), 2)
                    center_label_x, center_label_y = int(x), int(y)
                    cv2.putText(warped, label, (center_label_x, center_label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    results.append({
                        "width_cm": round(width_cm, 2),
                        "height_cm": round(height_cm, 2),
                    })

        # Save processed image
        output_filename = "processed_" + filename
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        cv2.imwrite(output_path, warped)

        # Process image info for database
        home_processed_image(output_filename)

        return output_filename, results

    except Exception as e:
        print(f"Error in detect_objects_from_camera: {str(e)}")
        return None, []

@app.route('/', methods=['GET', 'POST'])
def home():
    image_url = None
    measurements = []
    processed_filename = None
    file_exists = False

    # Check if we have camera results in session
    if 'camera_results' in session:
        camera_data = session.pop('camera_results', None)
        if camera_data:
            image_url = camera_data.get('image_url')
            measurements = camera_data.get('measurements', [])
            processed_filename = camera_data.get('processed_filename')

    if request.method == 'POST':
        # Check if it's a camera capture
        if 'camera_image' in request.form:
            camera_image = request.form['camera_image']
            filename = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Process the camera image
            processed_filename, measurements = detect_objects_from_camera(camera_image, filename)
            
            if processed_filename:
                image_url = '/' + os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                # Store results in session for display
                session['camera_results'] = {
                    'image_url': image_url,
                    'measurements': measurements,
                    'processed_filename': processed_filename
                }
                flash("Image captured and processed successfully!", "success")
                return redirect(url_for('home'))
            else:
                flash("Error processing captured image.", "error")
        
        # Legacy file upload (keep for compatibility)
        elif 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                if os.path.exists(file_path):
                    flash("Error: A file with this name already exists. Please rename the file or upload a different one.", "error")
                    file_exists = True
                else:
                    file.save(file_path)
                    home_processed_image(filename)
                    processed_filename, measurements = detect_objects(file_path, filename)
                    
                    if processed_filename:
                        image_url = '/' + os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

    return render_template('home.html', 
                         image_url=image_url, 
                         measurements=measurements, 
                         processed_filename=processed_filename, 
                         file_exists=file_exists)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """API endpoint for camera capture"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data received'})

        image_data = data['image']
        filename = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Process the image
        processed_filename, measurements = detect_objects_from_camera(image_data, filename)
        
        if processed_filename:
            image_url = '/' + os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            # Store in session for the next page load
            session['camera_results'] = {
                'image_url': image_url,
                'measurements': measurements,
                'processed_filename': processed_filename
            }
            
            return jsonify({
                'success': True,
                'redirect': url_for('home')
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to process image'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Keep the original detect_objects function for file uploads
def detect_objects(path, filename):
    image = cv2.imread(path)
    if image is None:
        return None, []
    image = cv2.resize(image, (1200, 1600))
    paper_pts = np.array([[0, 0], [1200, 0], [0, 1600], [1200, 1600]], dtype="float32")
    output_size = (int(A4_WIDTH_CM * PIXELS_PER_CM), int(A4_HEIGHT_CM * PIXELS_PER_CM))
    dst_pts = np.array([[0, 0],
                        [output_size[0] - 1, 0],
                        [0, output_size[1] - 1],
                        [output_size[0] - 1, output_size[1] - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(paper_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, output_size)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            if np.all(box[:, 0] >= 5) and np.all(box[:, 0] <= output_size[0] - 5) and \
               np.all(box[:, 1] >= 5) and np.all(box[:, 1] <= output_size[1] - 5):

                width_cm = w / PIXELS_PER_CM
                height_cm = h / PIXELS_PER_CM
                label = f"{width_cm:.1f}x{height_cm:.1f} cm"

                cv2.drawContours(warped, [box], 0, (0, 255, 0), 2)
                center_label_x, center_label_y = int(x), int(y)
                cv2.putText(warped, label, (center_label_x, center_label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                results.append({
                    "width_cm": round(width_cm, 2),
                    "height_cm": round(height_cm, 2),
                })

    output_filename = "processed_"+filename
    home_processed_image(output_filename)
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    
    cv2.imwrite(output_path, warped)

    return output_filename, results

@app.route('/save_measurements', methods=['POST'])
def save_measurements():
    try:
        image_filename = request.form.get('image_filename')
        if not image_filename:
            flash("Error: Image filename is missing.", "error")
            return redirect(url_for('home'))

        i = 1
        while True:
            object_name = request.form.get(f'object_name_{i}')
            width_cm = request.form.get(f'width_cm_{i}')
            height_cm = request.form.get(f'height_cm_{i}')
            if not object_name:
                break
            insert_object_measurements(image_filename, object_name, height_cm, width_cm)
            i += 1

        flash("Object saved successfully in Database!", "success")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f"object saving Error: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/image_info', methods=['GET', 'POST'])
def image_info():
    try:
        def to_float(val):
            try:
                if isinstance(val, numbers.Number):
                    return float(val)
                return float(val.numerator) / float(val.denominator)
            except Exception:
                return 72.0 

        if request.method == 'POST':
            image = request.files.get('image')
            if image:
                filename = image.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                if os.path.exists(filepath):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                image.save(filepath)

                img = Image.open(filepath)
                img_cv = cv2.imread(filepath)
                avg_color = img_cv.mean(axis=0).mean(axis=0)

                dpi_raw = img.info.get("dpi", (72, 72))
                dpi_x = to_float(dpi_raw[0])
                dpi_y = to_float(dpi_raw[1])

                info = {
                    'filename': filename,
                    'format': img.format,
                    'width': img.width,
                    'height': img.height,
                    'color_mode': img.mode,
                    'bit_depth': len(img.getbands()) * 8,
                    'compression': img.info.get("compression", "N/A"),
                    'dpi_x': round(dpi_x, 2),
                    'dpi_y': round(dpi_y, 2),
                    'physical_width_cm': round((img.width / dpi_x) * 2.54, 2),
                    'physical_height_cm': round((img.height / dpi_y) * 2.54, 2),
                    'avg_r': round(avg_color[2], 2),
                    'avg_g': round(avg_color[1], 2),
                    'avg_b': round(avg_color[0], 2),
                    'uploaded_at': datetime.now().strftime("%d %B %Y, %I:%M %p"),
                    'image_path': filename,
                    'size': os.path.getsize(filepath)
                }

                insert_image_info_mysql(info)

                return render_template('image_info.html', uploaded=True, **info)

        return render_template('image_info.html', uploaded=False)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/img_search')
def search():
    q = request.args.get('q', '').strip().lower()
    search_type = request.args.get('search_type', 'images')
    result = None
    results = []

    if q:
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            if search_type == 'image':
                query = """
                    SELECT * FROM images 
                    WHERE filename LIKE %s 
                    ORDER BY uploaded_at DESC 
                    LIMIT 1
                """
                cursor.execute(query, (f"%{q}%",))
                row = cursor.fetchone()

                if row:
                    result = {
                        'filename': row['filename'],
                        'format': row['format'],
                        'width': row['width'],
                        'height': row['height'],
                        'color_mode': row['color_mode'],
                        'bit_depth': row['bit_depth'],
                        'compression': row.get('compression', 'N/A'),
                        'dpi_x': row.get('dpi_x', 'N/A'),
                        'dpi_y': row.get('dpi_y', 'N/A'),
                        'physical_width_cm': round(row.get('physical_width_cm', 0), 2),
                        'physical_height_cm': round(row.get('physical_height_cm', 0), 2),
                        'avg_r': row.get('avg_r', 0),
                        'avg_g': row.get('avg_g', 0),
                        'avg_b': row.get('avg_b', 0),
                        'size': row.get('size', 0),
                        'uploaded_at': row['uploaded_at'].strftime("%d %B %Y, %I:%M %p"),
                        'image_file': url_for('uploaded_file', filename=row['filename'])
                    }
            elif search_type == 'object':
                query = """
                    SELECT * FROM object_measurements 
                    WHERE LOWER(object_name) LIKE %s 
                    OR LOWER(image_filename) LIKE %s 
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (f"%{q}%", f"%{q}%"))
                rows = cursor.fetchall()

                for row in rows:
                    results.append({
                        'image_filename': row['image_filename'],
                        'object_name': row['object_name'],
                        'height_cm': round(row['height_cm'], 2),
                        'width_cm': round(row['width_cm'], 2),
                        'created_at': row['created_at'].strftime("%d %B %Y, %I:%M %p")
                    })

            cursor.close()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}", 500

    return render_template('img_search.html', result=result, results=results, query=q, search_type=search_type)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)