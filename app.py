from flask import Flask, render_template, jsonify, request, Response
import sys
import os
import io
import json
import time
import numpy as np
import subprocess
from contextlib import redirect_stdout

# Add project root and tests directory to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tests'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from tests.test_gaia_data import run_and_visualize_gaia
from utils.download_gaia_data import download_gaia_subset
from tests.test_wine_dataset import run_and_visualize_wine
from src.streaming_simulator import StreamingSensorSimulator
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

app = Flask(__name__)

# Global progress tracking
analysis_progress = {
    'status': 'idle',
    'messages': [],
    'current_message': '',
    'start_time': None
}


def _get_nystrom_enabled(request_data):
    """Return whether the experimental Nystrom approximation flag is enabled."""
    if not request_data:
        return False
    raw_value = request_data.get('nystrom_enabled', False)
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in ('1', 'true', 'yes', 'on')


def _log_nystrom_status(nystrom_enabled):
    """Log the current Nystrom approximation mode."""
    if nystrom_enabled:
        print("Nystrom approximation: requested, planned next-step mode. Full-kernel path still runs.")
    else:
        print("Nystrom approximation: off, using full-kernel path.")


@app.route('/')
def index():
    """
    Serves the main page of the application.
    """
    return render_template('index.html')

@app.route('/pancreas-analysis')
def pancreas_analysis():
    """
    Serves the pancreas single-cell RNA-seq analysis page.
    """
    return render_template('pancreas_analysis.html')

@app.route('/biological-metrics')
def biological_metrics():
    """
    Serves the biological metrics analysis page.
    """
    return render_template('biological_metrics.html')

@app.route('/smart-sampling')
def smart_sampling():
    """
    Serves the smart sampling analysis page.
    """
    return render_template('smart_sampling.html')

@app.route('/gaia-analysis')
def gaia_analysis():
    """
    Serves the GAIA dataset analysis page.
    """
    return render_template('gaia_analysis.html')

@app.route('/run', methods=['POST'])
def run_analysis():
    """
    Runs the Gaia data analysis and returns the results, including logs.
    """
    request_data = request.get_json(silent=True) or {}
    size = request_data.get('size', '500')
    kernel_type = request_data.get('kernel_type', 'exponential')
    kernel_p = request_data.get('kernel_p', 1.5)
    kernel_nu = request_data.get('kernel_nu', 1.0)
    kernel_alpha = request_data.get('kernel_alpha', 1.0)
    kernel_beta = request_data.get('kernel_beta', 1.0)
    learn_kernel_beta = request_data.get('learn_kernel_beta', False)
    k_adaptation_strategy = request_data.get('k_adaptation_strategy', 'off')
    k_base = request_data.get('k_base', 20)
    sampling_method = request_data.get('sampling_method', 'off')
    target_size = request_data.get('target_size', 2000)
    spatial_weight = request_data.get('spatial_weight', 0.7)
    nystrom_enabled = _get_nystrom_enabled(request_data)
    size_int = int(size)

    data_dir = "data"
    filename = f"gaia_data_{size}.csv"
    data_path = os.path.join(data_dir, filename)

    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        _log_nystrom_status(nystrom_enabled)
        # Check if data exists, download if not
        if not os.path.exists(data_path):
            print(f"Data file '{data_path}' not found. Downloading...")
            try:
                download_gaia_subset(limit=size_int, filename=filename)
            except Exception as e:
                error_message = f"Failed to download data. Error: {e}"
                print(error_message)
                # Still return the logs so user sees the download attempt
                return jsonify({'success': False, 'error': error_message, 'logs': log_stream.getvalue()})

        # Run the main analysis
        image_path, timings = run_and_visualize_gaia(
            data_path=data_path,
            kernel_type=kernel_type,
            kernel_p=kernel_p,
            kernel_nu=kernel_nu,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            learn_kernel_beta=learn_kernel_beta,
            k_adaptation_strategy=k_adaptation_strategy,
            k_base=k_base,
            sampling_method=sampling_method,
            target_size=target_size,
            spatial_weight=spatial_weight,
        )
    
    logs = log_stream.getvalue()

    if image_path:
        return jsonify({
            'success': True, 
            'image_path': image_path.replace('static' + os.path.sep, '', 1), 
            'timings': timings,
            'sample_count': size,
            'nystrom_enabled': nystrom_enabled,
            'logs': logs
        })
    else:
        error = timings.get('error', 'An unknown error occurred during analysis.')
        return jsonify({'success': False, 'error': error, 'logs': logs})

@app.route('/run_wine', methods=['POST'])
def run_wine_analysis():
    """
    Runs the Wine dataset analysis and returns the results.
    """
    # Get kernel configuration from request data
    request_data = request.get_json() or {}
    kernel_type = request_data.get('kernel_type', 'exponential')
    kernel_p = request_data.get('kernel_p', 1.5)
    kernel_nu = request_data.get('kernel_nu', 1.0)
    kernel_alpha = request_data.get('kernel_alpha', 1.0)
    kernel_beta = request_data.get('kernel_beta', 1.0)
    learn_kernel_beta = request_data.get('learn_kernel_beta', False)
    k_adaptation_strategy = request_data.get('k_adaptation_strategy', 'off')
    k_base = request_data.get('k_base', 20)
    sampling_method = request_data.get('sampling_method', 'off')
    target_size = request_data.get('target_size', 150)
    spatial_weight = request_data.get('spatial_weight', 0.7)
    nystrom_enabled = _get_nystrom_enabled(request_data)

    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        _log_nystrom_status(nystrom_enabled)
        image_path, timings = run_and_visualize_wine(
            kernel_type=kernel_type,
            kernel_p=kernel_p,
            kernel_nu=kernel_nu,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            learn_kernel_beta=learn_kernel_beta,
            k_adaptation_strategy=k_adaptation_strategy,
            k_base=k_base,
            sampling_method=sampling_method,
            target_size=target_size,
            spatial_weight=spatial_weight,
        )
    
    logs = log_stream.getvalue()

    if image_path:
        # Normalize path separators to forward slashes for web
        normalized_path = image_path.replace(os.path.sep, '/')
        # Remove 'static/' prefix if present to avoid duplication
        if normalized_path.startswith('static/'):
            normalized_path = normalized_path[7:]
        
        return jsonify({
            'success': True, 
            'image_path': normalized_path,
            'timings': timings,
            'sample_count': '178',  # Wine dataset size
            'nystrom_enabled': nystrom_enabled,
            'logs': logs
        })
    else:
        error = timings.get('error', 'An unknown error occurred during analysis.')
        return jsonify({'success': False, 'error': error, 'logs': logs})

@app.route('/run_pancreas', methods=['POST'])
def run_pancreas_analysis():
    """
    Runs the Pancreas endocrinogenesis single-cell RNA-seq analysis and returns the results.
    """
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        try:
            from tests.test_pancreas_endocrinogenesis import run_and_visualize_pancreas

            # Get kernel configuration from request data
            request_data = request.get_json() or {}
            kernel_type = request_data.get('kernel_type', 'exponential')
            kernel_p = request_data.get('kernel_p', 1.5)
            kernel_nu = request_data.get('kernel_nu', 1.0)
            kernel_alpha = request_data.get('kernel_alpha', 1.0)
            kernel_beta = request_data.get('kernel_beta', 1.0)
            learn_kernel_beta = request_data.get('learn_kernel_beta', False)
            k_adaptation_strategy = request_data.get('k_adaptation_strategy', 'off')
            k_base = request_data.get('k_base', 20)
            sampling_method = request_data.get('sampling_method', 'off')
            target_size = request_data.get('target_size', 2000)
            spatial_weight = request_data.get('spatial_weight', 0.7)
            nystrom_enabled = _get_nystrom_enabled(request_data)
            _log_nystrom_status(nystrom_enabled)

            image_path, timings = run_and_visualize_pancreas(
                kernel_type=kernel_type,
                kernel_p=kernel_p,
                kernel_nu=kernel_nu,
                kernel_alpha=kernel_alpha,
                kernel_beta=kernel_beta,
                learn_kernel_beta=learn_kernel_beta,
                k_adaptation_strategy=k_adaptation_strategy,
                k_base=k_base,
                sampling_method=sampling_method,
                target_size=target_size,
                spatial_weight=spatial_weight,
            )
        except Exception as e:
            error_message = f"Analysis failed: {str(e)}"
            print(error_message)
            return jsonify({'success': False, 'error': error_message, 'logs': log_stream.getvalue()})
    
    logs = log_stream.getvalue()

    if image_path:
        # Remove 'static/' or 'static\' prefix if present, then ensure forward slashes
        clean_path = image_path
        if clean_path.startswith('static/'):
            clean_path = clean_path[7:]  # Remove 'static/'
        elif clean_path.startswith('static\\'):
            clean_path = clean_path[8:]  # Remove 'static\'
        clean_path = clean_path.replace('\\', '/')
        
        return jsonify({
            'success': True, 
            'image_path': clean_path, 
            'timings': timings,
            'sample_count': 'Single-Cell RNA-seq',
            'nystrom_enabled': nystrom_enabled,
            'logs': logs
        })
    else:
        error = timings.get('error', 'An unknown error occurred during analysis.')
        return jsonify({'success': False, 'error': error, 'logs': logs})

# Global variables for streaming demo
streaming_simulator = None
streaming_algorithm = None
social_media_streaming = False  # Simple flag to control social media stream

@app.route('/streaming-demo')
def streaming_demo():
    """
    Serves the streaming demo page.
    """
    return render_template('streaming_demo.html')

@app.route('/api/streaming-data')
def streaming_data():
    """
    Server-Sent Events endpoint for real-time streaming data.
    
    Query Parameters:
        kernel_type: 'exponential' (default) or 'gaussian'
            - 'exponential': K = exp(-d / (2σ²)) - linear distance decay, empirically effective
            - 'gaussian': K = exp(-d² / (2σ²)) - squared distance decay, standard RBF
    """
    global streaming_simulator, streaming_algorithm
    
    # Get kernel configuration from query parameters
    kernel_type = request.args.get('kernel_type', 'exponential')
    kernel_p_raw = request.args.get('kernel_p', None)
    kernel_nu_raw = request.args.get('kernel_nu', None)
    kernel_alpha_raw = request.args.get('kernel_alpha', None)
    kernel_beta_raw = request.args.get('kernel_beta', None)
    learn_kernel_beta_raw = request.args.get('learn_kernel_beta', None)

    # Parse numeric kernel parameters (optional)
    try:
        kernel_p = float(kernel_p_raw) if kernel_p_raw is not None else 1.5
    except Exception:
        kernel_p = 1.5
    try:
        kernel_nu = float(kernel_nu_raw) if kernel_nu_raw is not None else 1.0
    except Exception:
        kernel_nu = 1.0
    try:
        kernel_alpha = float(kernel_alpha_raw) if kernel_alpha_raw is not None else 1.0
    except Exception:
        kernel_alpha = 1.0
    try:
        kernel_beta = float(kernel_beta_raw) if kernel_beta_raw is not None else 1.0
    except Exception:
        kernel_beta = 1.0

    if learn_kernel_beta_raw is None:
        learn_kernel_beta = False
    else:
        learn_kernel_beta = str(learn_kernel_beta_raw).strip().lower() in ('1', 'true', 'yes', 'on')

    valid_kernel_types = ('exponential', 'gaussian', 'generalized', 'student_t', 'rational_quadratic')
    if kernel_type not in valid_kernel_types:
        kernel_type = 'exponential'
    
    def generate():
        # Initialize simulator and algorithm with selected kernel configuration
        simulator = StreamingSensorSimulator(n_sensors=6, update_interval=0.1)
        algorithm = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=10,
            device='cpu',
            kernel_type=kernel_type,
            kernel_p=kernel_p,
            kernel_nu=kernel_nu,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            learn_kernel_beta=learn_kernel_beta,
        )
        
        point_count = 0
        
        try:
            for reading in simulator.stream():
                start_time = time.time()
                
                # Check for pending anomaly injections
                if hasattr(inject_anomaly, 'pending_anomalies') and inject_anomaly.pending_anomalies:
                    for anomaly_request in inject_anomaly.pending_anomalies:
                        simulator.inject_anomaly(
                            anomaly_type=anomaly_request['type'],
                            sensor_name=anomaly_request['sensor'],
                            duration=anomaly_request['duration'],
                            intensity=anomaly_request['intensity']
                        )
                    inject_anomaly.pending_anomalies.clear()
                
                # Check for clear anomaly requests
                if hasattr(streaming_control, 'clear_requests') and streaming_control.clear_requests:
                    simulator.clear_anomalies()
                    streaming_control.clear_requests.clear()
                
                # Get sensor values as numpy array
                sensor_values = np.array(reading['values_array'])
                
                # Update embedding
                embedding = algorithm.update_embedding(sensor_values)
                
                # Calculate update time
                update_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Prepare data for frontend
                data = {
                    'embedding': embedding[-1].tolist() if len(embedding) > 0 else [0, 0],  # Latest point
                    'all_embeddings': embedding.tolist(),  # All points for visualization
                    'sensor_reading': reading,
                    'update_time': round(update_time, 2),
                    'point_count': point_count,
                    'timestamp': reading['timestamp'],
                    'anomaly_status': reading.get('anomaly_status', {}),
                    'has_anomaly': reading.get('has_anomaly', False)
                }
                
                # Send as Server-Sent Event
                yield f"data: {json.dumps(data)}\n\n"
                
                point_count += 1
                
                # Stop after 1000 points to prevent infinite streaming
                if point_count >= 1000:
                    break
                    
        except Exception as e:
            error_data = {'error': str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response

@app.route('/api/streaming-control', methods=['POST'])
def streaming_control():
    """
    Control endpoint for streaming demo (start/stop/reset).
    """
    action = request.json.get('action', '')
    
    if action == 'reset':
        # Reset the algorithm state
        global streaming_algorithm
        if streaming_algorithm:
            streaming_algorithm.reset_streaming()
        return jsonify({'success': True, 'message': 'Streaming reset'})
    
    elif action == 'clear_anomalies':
        # Store the clear request for the streaming generator
        if not hasattr(streaming_control, 'clear_requests'):
            streaming_control.clear_requests = []
        streaming_control.clear_requests.append(True)
        return jsonify({'success': True, 'message': 'Anomalies cleared'})
    
    return jsonify({'success': True, 'message': f'Action {action} acknowledged'})

@app.route('/api/inject-anomaly', methods=['POST'])
def inject_anomaly():
    """
    Inject anomalies into the streaming data.
    """
    global streaming_simulator
    
    data = request.json
    anomaly_type = data.get('type', 'spike')
    sensor_name = data.get('sensor', None)
    duration = data.get('duration', 50)
    intensity = data.get('intensity', 1.0)
    
    # We need to get the current simulator instance from the streaming
    # For now, we'll store the anomaly request and apply it in the next stream
    anomaly_request = {
        'type': anomaly_type,
        'sensor': sensor_name,
        'duration': duration,
        'intensity': intensity,
        'timestamp': time.time()
    }
    
    # Store in a global variable that the streaming generator can access
    if not hasattr(inject_anomaly, 'pending_anomalies'):
        inject_anomaly.pending_anomalies = []
    inject_anomaly.pending_anomalies.append(anomaly_request)
    
    return jsonify({
        'success': True, 
        'message': f'Anomaly {anomaly_type} injected',
        'anomaly': anomaly_request
    })

@app.route('/api/biological_metrics', methods=['POST'])
def run_biological_metrics():
    """
    Runs the biological metrics analysis and returns the results.
    """
    try:
        print("Initializing biological metrics analysis...")
        
        dataset = request.json.get('dataset', 'pancreas') if request.json else 'pancreas'
        demo_mode = request.json.get('demo_mode', True) if request.json else True  # Default to demo mode
        kernel_type = request.json.get('kernel_type', 'exponential') if request.json else 'exponential'
        kernel_p = request.json.get('kernel_p', 1.5) if request.json else 1.5
        kernel_nu = request.json.get('kernel_nu', 1.0) if request.json else 1.0
        kernel_alpha = request.json.get('kernel_alpha', 1.0) if request.json else 1.0
        kernel_beta = request.json.get('kernel_beta', 1.0) if request.json else 1.0
        learn_kernel_beta = request.json.get('learn_kernel_beta', False) if request.json else False
        k_adaptation_strategy = request.json.get('k_adaptation_strategy', 'off') if request.json else 'off'
        k_base = request.json.get('k_base', 20) if request.json else 20
        sampling_method = request.json.get('sampling_method', 'off') if request.json else 'off'
        target_size = request.json.get('target_size', 2000) if request.json else 2000
        spatial_weight = request.json.get('spatial_weight', 0.7) if request.json else 0.7
        selected_algorithms = request.json.get('algorithms', None) if request.json else None
        nystrom_enabled = _get_nystrom_enabled(request.json)
        print(f"Running analysis for dataset: {dataset}")
        print(f"Computation mode: {'Fast demo mode' if demo_mode else 'Full scientific mode'}")
        print(f"Kernel type: {kernel_type}")
        _log_nystrom_status(nystrom_enabled)
        
        # Validate dataset choice
        if dataset not in ['pancreas', 'synthetic', 'bodenmiller']:
            return jsonify({
                'status': 'error',
                'error': f'Unsupported dataset: {dataset}. Supported options: pancreas, synthetic, bodenmiller',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }), 400
        
        # Handle Bodenmiller dataset (not yet implemented)
        if dataset == 'bodenmiller':
            return jsonify({
                'status': 'error',
                'error': 'Bodenmiller CyTOF dataset is not yet implemented. Please choose pancreas or synthetic.',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }), 501
        
        # Import here to avoid circular imports - try enhanced metrics first
        try:
            from tests.test_enhanced_biological_metrics import run_enhanced_biological_metrics_comparison
            use_enhanced = True
        except ImportError:
            from tests.test_biological_metrics import run_biological_metrics_comparison
            use_enhanced = False
        
        log_stream = io.StringIO()
        
        with redirect_stdout(log_stream):
            # Run the appropriate analysis based on dataset
            image_path = None
            if use_enhanced:
                print("Executing enhanced biological metrics comparison...")
                if dataset == 'synthetic':
                    print("Using synthetic benchmark dataset...")
                    results, embeddings, image_path = run_enhanced_biological_metrics_comparison(
                        use_synthetic=True,
                        demo_mode=demo_mode,
                        kernel_type=kernel_type,
                        kernel_p=kernel_p,
                        kernel_nu=kernel_nu,
                        kernel_alpha=kernel_alpha,
                        kernel_beta=kernel_beta,
                        learn_kernel_beta=learn_kernel_beta,
                        k_adaptation_strategy=k_adaptation_strategy,
                        k_base=k_base,
                        sampling_method=sampling_method,
                        target_size=target_size,
                        spatial_weight=spatial_weight,
                        selected_algorithms=selected_algorithms,
                    )
                else:  # pancreas
                    print("Using real pancreas endocrinogenesis dataset...")
                    results, embeddings, image_path = run_enhanced_biological_metrics_comparison(
                        use_synthetic=False,
                        demo_mode=demo_mode,
                        kernel_type=kernel_type,
                        kernel_p=kernel_p,
                        kernel_nu=kernel_nu,
                        kernel_alpha=kernel_alpha,
                        kernel_beta=kernel_beta,
                        learn_kernel_beta=learn_kernel_beta,
                        k_adaptation_strategy=k_adaptation_strategy,
                        k_base=k_base,
                        sampling_method=sampling_method,
                        target_size=target_size,
                        spatial_weight=spatial_weight,
                        selected_algorithms=selected_algorithms,
                    )
            else:
                if dataset == 'synthetic':
                    print("Synthetic dataset not supported in standard metrics - using pancreas data...")
                print("Executing standard biological metrics comparison...")
                results, embeddings, image_path = run_biological_metrics_comparison()
            print("Analysis completed successfully.")
        
        # Get the captured output
        output = log_stream.getvalue()
        
        # Format results for JSON response (convert NumPy types to Python native types)
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            else:
                return obj
        
        formatted_results = {}
        for alg_name, alg_results in results.items():
            if 'biological_metrics' in alg_results:
                formatted_results[alg_name] = {
                    'runtime': float(alg_results['runtime']),
                    'biological_metrics': convert_numpy_types(alg_results['biological_metrics'])
                }
            else:
                formatted_results[alg_name] = convert_numpy_types(alg_results)
        
        print(f"Returning results for {len(formatted_results)} algorithms")
        
        # Clean up image path for web serving
        web_image_path = None
        if image_path:
            # Convert absolute path to relative web path
            if image_path.startswith('static/'):
                web_image_path = image_path[7:]  # Remove 'static/' prefix
            elif image_path.startswith('static\\'):
                web_image_path = image_path[8:].replace('\\', '/')  # Remove 'static\' and fix slashes
            else:
                # Extract just the filename if it's a full path
                web_image_path = 'results/' + os.path.basename(image_path)

        return jsonify({
            'status': 'success',
            'results': formatted_results,
            'dataset': dataset,
            'nystrom_enabled': nystrom_enabled,
            'output': output,
            'image_path': web_image_path,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in biological metrics analysis: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


@app.route('/synthetic-developmental')
def synthetic_developmental():
    """
    Render the synthetic developmental trajectories page.
    """
    return render_template('synthetic_developmental.html')

@app.route('/mouse-brain-cortical')
def mouse_brain_cortical():
    """
    Render the mouse brain cortical layers analysis page.
    """
    return render_template('mouse_brain_cortical.html')


@app.route('/api/synthetic-developmental/progress', methods=['GET'])
def get_synthetic_progress():
    """
    Get current progress of synthetic developmental analysis.
    """
    return jsonify(analysis_progress)

@app.route('/api/synthetic-developmental', methods=['POST'])
def run_synthetic_developmental():
    """
    Run synthetic developmental trajectory analysis and return results.
    """
    try:
        print("Initializing synthetic developmental analysis...")
        
        # Import here to avoid circular imports
        from tests.test_synthetic_developmental import run_synthetic_developmental_comparison
        from tests.synthetic_developmental_datasets import create_all_synthetic_datasets
        
        dataset_name = request.json.get('dataset', 'hematopoietic') if request.json else 'hematopoietic'
        kernel_type = request.json.get('kernel_type', 'exponential') if request.json else 'exponential'
        kernel_p = request.json.get('kernel_p', 1.5) if request.json else 1.5
        kernel_nu = request.json.get('kernel_nu', 1.0) if request.json else 1.0
        kernel_alpha = request.json.get('kernel_alpha', 1.0) if request.json else 1.0
        kernel_beta = request.json.get('kernel_beta', 1.0) if request.json else 1.0
        learn_kernel_beta = request.json.get('learn_kernel_beta', False) if request.json else False
        k_adaptation_strategy = request.json.get('k_adaptation_strategy', 'off') if request.json else 'off'
        k_base = request.json.get('k_base', 20) if request.json else 20
        sampling_method = request.json.get('sampling_method', 'off') if request.json else 'off'
        target_size = request.json.get('target_size', 2000) if request.json else 2000
        spatial_weight = request.json.get('spatial_weight', 0.7) if request.json else 0.7
        selected_algorithms = request.json.get('algorithms', None) if request.json else None
        nystrom_enabled = _get_nystrom_enabled(request.json)
        print(f"Running analysis for dataset: {dataset_name}")
        print(f"Kernel type: {kernel_type}")
        _log_nystrom_status(nystrom_enabled)
        
        # Initialize progress tracking
        global analysis_progress
        analysis_progress = {
            'status': 'running',
            'messages': [],
            'current_message': 'Starting analysis...',
            'start_time': time.time()
        }
        
        # Create a custom stdout capture that also updates progress
        log_stream = io.StringIO()
        
        class ProgressStream:
            def __init__(self, original_stream, log_stream):
                self.original_stream = original_stream
                self.log_stream = log_stream
                self.buffer = ""
            
            def write(self, data):
                global analysis_progress
                self.original_stream.write(data)
                self.log_stream.write(data)
                
                # Update progress with new lines
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        analysis_progress['messages'].append(line.strip())
                        analysis_progress['current_message'] = line.strip()
                        # Keep only last 100 messages to prevent memory issues
                        if len(analysis_progress['messages']) > 100:
                            analysis_progress['messages'] = analysis_progress['messages'][-100:]
                
                self.original_stream.flush()
                self.log_stream.flush()
            
            def flush(self):
                self.original_stream.flush()
                self.log_stream.flush()
        
        # Redirect stdout to our progress-aware stream
        progress_stream = ProgressStream(sys.stdout, log_stream)
        
        with redirect_stdout(progress_stream):
            # Generate the specific dataset requested
            print("Generating synthetic developmental datasets...")
            all_datasets = create_all_synthetic_datasets(random_seed=42)
            
            # Filter to the requested dataset
            if dataset_name in all_datasets:
                selected_datasets = {dataset_name: all_datasets[dataset_name]}
            else:
                # Default to hematopoietic if not found
                selected_datasets = {'hematopoietic': all_datasets['hematopoietic']}
                dataset_name = 'hematopoietic'
            
            print(f"Executing synthetic developmental comparison for {dataset_name}...")
            results, embeddings, image_path = run_synthetic_developmental_comparison(
                datasets=selected_datasets,
                save_results=True,
                kernel_type=kernel_type,
                kernel_p=kernel_p,
                kernel_nu=kernel_nu,
                kernel_alpha=kernel_alpha,
                kernel_beta=kernel_beta,
                learn_kernel_beta=learn_kernel_beta,
                k_adaptation_strategy=k_adaptation_strategy,
                k_base=k_base,
                sampling_method=sampling_method,
                target_size=target_size,
                spatial_weight=spatial_weight,
                selected_algorithms=selected_algorithms,
            )
            print("Synthetic analysis completed successfully.")
        
        # Get the captured output
        output = log_stream.getvalue()
        
        # Helper function to convert numpy types to JSON serializable types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Format results for frontend
        dataset_info = selected_datasets[dataset_name]
        formatted_results = {}
        formatted_embeddings = {}
        formatted_metrics = {}
        
        if dataset_name in results:
            dataset_results = results[dataset_name]
            
            for algorithm, result in dataset_results.items():
                if result is not None:
                    # Format evaluation results
                    formatted_results[algorithm] = convert_numpy_types({
                        'runtime': result['runtime'],
                        'evaluation': result['evaluation']
                    })
                    
                    # Format embeddings (convert numpy arrays to lists)
                    if dataset_name in embeddings and algorithm in embeddings[dataset_name]:
                        embedding = embeddings[dataset_name][algorithm]
                        formatted_embeddings[algorithm] = embedding.tolist()
                    
                    # Store metrics for visualization
                    formatted_metrics[algorithm] = convert_numpy_types(result)
        
        # Format ground truth information
        ground_truth = convert_numpy_types({
            'spatial_coordinates': dataset_info['spatial_coordinates'],
            'pseudotime': dataset_info['true_pseudotime'],
            'cell_types': dataset_info['cell_types']
        })
        
        # Dataset information
        dataset_info_formatted = {
            'n_cells': int(dataset_info['X'].shape[0]),
            'n_features': int(dataset_info['X'].shape[1]), 
            'n_cell_types': int(len(np.unique(dataset_info['cell_types']))),
            'complexity': 'High' if 'neural_crest' in dataset_name else 'Medium',
            'description': dataset_info['description']
        }
        
        # Extract just the filename from the full path for the frontend
        image_filename = os.path.basename(image_path) if image_path else None
        
        # Update progress status to completed
        analysis_progress['status'] = 'completed'
        analysis_progress['current_message'] = 'Analysis completed successfully!'
        
        return jsonify({
            'status': 'success',
            'results': formatted_results,
            'embeddings': formatted_embeddings,
            'metrics': formatted_metrics,
            'ground_truth': ground_truth,
            'dataset_info': dataset_info_formatted,
            'dataset_name': dataset_name,
            'nystrom_enabled': nystrom_enabled,
            'output': output,
            'image_path': f"results/{image_filename}" if image_filename else None,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in synthetic developmental analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Update progress status to error
        analysis_progress['status'] = 'error'
        analysis_progress['current_message'] = f'Error: {str(e)}'
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


@app.route('/api/smart_sampling/progress', methods=['GET'])
def get_smart_sampling_progress():
    """
    Get current progress of smart sampling analysis.
    """
    return jsonify(analysis_progress)

@app.route('/api/smart_sampling/quick_test', methods=['POST'])
def run_smart_sampling_quick_test():
    """
    Run the simple quick test that shows terminal output exactly as in src/test_smart_sampling_quick.py
    """
    try:
        print("[ROCKET] Testing smart sampling analysis...")
        
        # Initialize progress tracking
        global analysis_progress
        analysis_progress = {
            'status': 'running',
            'messages': [],
            'current_message': 'Starting smart sampling test...',
            'start_time': time.time()
        }
        
        # Create a custom stdout capture that also updates progress
        log_stream = io.StringIO()
        
        class ProgressStream:
            def __init__(self, original_stream, log_stream):
                self.original_stream = original_stream
                self.log_stream = log_stream
                self.buffer = ""
            
            def write(self, data):
                global analysis_progress
                self.original_stream.write(data)
                self.log_stream.write(data)
                
                # Update progress with new lines
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        analysis_progress['messages'].append(line.strip())
                        analysis_progress['current_message'] = line.strip()
                        # Keep only last 1000 messages to show the full analysis
                        if len(analysis_progress['messages']) > 1000:
                            analysis_progress['messages'] = analysis_progress['messages'][-1000:]
                
                self.original_stream.flush()
                self.log_stream.flush()
            
            def flush(self):
                self.original_stream.flush()
                self.log_stream.flush()
        
        # Redirect stdout to our progress-aware stream
        progress_stream = ProgressStream(sys.stdout, log_stream)
        
        with redirect_stdout(progress_stream):
            print("[ROCKET] Testing smart sampling analysis...")
            
            try:
                # Import the function
                from tests.smart_sampling_enhanced_analysis import run_smart_sampling_analysis
                
                print("[CHECK] Successfully imported smart sampling function")
                print("[CHART] Starting analysis...")
                
                # Run with timing
                start_time = time.time()
                
                # Run the analysis
                results, embeddings, sampling_info = run_smart_sampling_analysis()
                
                end_time = time.time()
                
                print(f"[CHECK] Analysis completed in {end_time - start_time:.1f}s")
                print(f"[CHART] Results keys: {list(results.keys()) if results else 'None'}")
                print(f"[CHART] Embeddings keys: {list(embeddings.keys()) if embeddings else 'None'}")
                print(f"[CHART] Sampling info keys: {list(sampling_info.keys()) if sampling_info else 'None'}")
                
                print("[PARTY] Smart sampling test completed successfully!")
                
            except Exception as e:
                print(f"[ERROR] Error during analysis: {e}")
                import traceback
                traceback.print_exc()
                print("[BOOM] Smart sampling test failed!")
        
        # Get the captured output
        output = log_stream.getvalue()
        
        # Update progress status to completed
        analysis_progress['status'] = 'completed'
        analysis_progress['current_message'] = 'Smart sampling test completed!'
        
        return jsonify({
            'status': 'success',
            'output': output,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in smart sampling quick test: {e}")
        import traceback
        traceback.print_exc()
        
        # Update progress status to error
        analysis_progress['status'] = 'error'
        analysis_progress['current_message'] = f'Error: {str(e)}'
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'output': traceback.format_exc(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


@app.route('/api/smart_sampling/run', methods=['POST'])
def run_smart_sampling_run():
    """
    Run smart sampling analysis with user-selected settings.
    """
    try:
        request_data = request.get_json(silent=True) or {}
        sampling_method = request_data.get('sampling_method', 'off')
        target_size = int(request_data.get('target_size', 2000))
        spatial_weight = float(request_data.get('spatial_weight', 0.7))

        print("[ROCKET] Running smart sampling analysis (configured)...")

        # Initialize progress tracking
        global analysis_progress
        analysis_progress = {
            'status': 'running',
            'messages': [],
            'current_message': 'Starting smart sampling analysis...',
            'start_time': time.time()
        }

        # Create a custom stdout capture that also updates progress
        log_stream = io.StringIO()

        class ProgressStream:
            def __init__(self, original_stream, log_stream):
                self.original_stream = original_stream
                self.log_stream = log_stream
                self.buffer = ""

            def write(self, data):
                global analysis_progress
                self.original_stream.write(data)
                self.log_stream.write(data)

                # Update progress with new lines
                self.buffer += data
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        analysis_progress['messages'].append(line.strip())
                        analysis_progress['current_message'] = line.strip()
                        if len(analysis_progress['messages']) > 1000:
                            analysis_progress['messages'] = analysis_progress['messages'][-1000:]

                self.original_stream.flush()
                self.log_stream.flush()

            def flush(self):
                self.original_stream.flush()
                self.log_stream.flush()

        progress_stream = ProgressStream(sys.stdout, log_stream)

        with redirect_stdout(progress_stream):
            from tests.smart_sampling_enhanced_analysis import run_smart_sampling_analysis

            run_smart_sampling_analysis(
                sampling_method=sampling_method,
                target_size=target_size,
                spatial_weight=spatial_weight,
            )

        output = log_stream.getvalue()

        # Update progress status to completed
        analysis_progress['status'] = 'completed'
        analysis_progress['current_message'] = 'Smart sampling analysis completed!'

        # Find the latest generated visualization files (same logic as /latest)
        results_dir = os.path.join('static', 'results')
        visualization_files = []

        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith('smart_sampling_') and filename.endswith('.png'):
                    visualization_files.append(filename)

        visualization_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(results_dir, f)),
            reverse=True
        )

        latest_analysis = None
        latest_performance = None
        latest_comparison = None

        for filename in visualization_files:
            if 'enhanced_analysis_' in filename and latest_analysis is None:
                latest_analysis = filename
            elif 'performance_chart_' in filename and latest_performance is None:
                latest_performance = filename
            elif 'comparison_' in filename and latest_comparison is None:
                latest_comparison = filename

        analysis_chart = latest_analysis or latest_comparison
        performance_chart = latest_performance

        return jsonify({
            'status': 'success',
            'output': output,
            'visualizations': {
                'analysis_chart': f'results/{analysis_chart}' if analysis_chart else None,
                'performance_chart': f'results/{performance_chart}' if performance_chart else None
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print(f"Error in smart sampling configured run: {e}")
        import traceback
        traceback.print_exc()

        # Update progress status to error
        analysis_progress['status'] = 'error'
        analysis_progress['current_message'] = f'Error: {str(e)}'

        return jsonify({
            'status': 'error',
            'error': str(e),
            'output': traceback.format_exc(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


@app.route('/api/smart_sampling/latest', methods=['GET'])
def get_latest_smart_sampling_results():
    """
    Get the latest smart sampling visualization results without running analysis.
    """
    try:
        # Find the latest generated visualization files
        results_dir = os.path.join('static', 'results')
        visualization_files = []
        
        if os.path.exists(results_dir):
            # Look for all smart sampling visualization files
            for filename in os.listdir(results_dir):
                if (filename.startswith('smart_sampling_') and filename.endswith('.png')):
                    visualization_files.append(filename)
        
        if not visualization_files:
            return jsonify({
                'status': 'error',
                'error': 'No previous smart sampling results found'
            }), 404
        
        # Sort by modification time (newest first)
        visualization_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
        
        # Get the most recent files by type
        latest_analysis = None
        latest_performance = None
        latest_comparison = None
        
        for filename in visualization_files:
            if 'enhanced_analysis_' in filename and latest_analysis is None:
                latest_analysis = filename
            elif 'performance_chart_' in filename and latest_performance is None:
                latest_performance = filename
            elif 'comparison_' in filename and latest_comparison is None:
                latest_comparison = filename
        
        # Use the most appropriate visualizations available
        analysis_chart = latest_analysis or latest_comparison
        performance_chart = latest_performance
        
        return jsonify({
            'status': 'success',
            'visualizations': {
                'analysis_chart': f'results/{analysis_chart}' if analysis_chart else None,
                'performance_chart': f'results/{performance_chart}' if performance_chart else None
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'note': 'Loaded latest available visualizations'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/mouse_brain_cortical', methods=['POST'])
def run_mouse_brain_cortical():
    """
    Run mouse brain cortical layers analysis and return results.
    """
    try:
        print("Initializing mouse brain cortical analysis...")
        
        # Import here to avoid circular imports
        from tests.test_mouse_brain_cortical import run_and_visualize_mouse_brain_cortical
        
        # Get request parameters
        request_data = request.get_json() or {}
        dataset = request_data.get('dataset', 'synthetic')
        n_cells = request_data.get('n_cells', 6000)
        n_layers = request_data.get('n_layers', 5)
        algorithms = request_data.get('algorithms', ['normdyn', 'tsne', 'umap'])
        metrics = request_data.get('metrics', ['gradient', 'continuity', 'boundaries', 'ordering'])
        kernel_type = request_data.get('kernel_type', 'exponential')
        kernel_p = request_data.get('kernel_p', 1.5)
        kernel_nu = request_data.get('kernel_nu', 1.0)
        kernel_alpha = request_data.get('kernel_alpha', 1.0)
        kernel_beta = request_data.get('kernel_beta', 1.0)
        learn_kernel_beta = request_data.get('learn_kernel_beta', False)
        k_adaptation_strategy = request_data.get('k_adaptation_strategy', 'off')
        k_base = request_data.get('k_base', 20)
        sampling_method = request_data.get('sampling_method', 'off')
        target_size = request_data.get('target_size', 2000)
        spatial_weight = request_data.get('spatial_weight', 0.7)
        nystrom_enabled = _get_nystrom_enabled(request_data)

        print(f"Parameters: dataset={dataset}, n_cells={n_cells}, n_layers={n_layers}, kernel_type={kernel_type}")
        print(f"Algorithms: {algorithms}")
        print(f"Metrics: {metrics}")
        _log_nystrom_status(nystrom_enabled)
        
        # Run the analysis
        image_path, results, metadata = run_and_visualize_mouse_brain_cortical(
            kernel_type=kernel_type,
            kernel_p=kernel_p,
            kernel_nu=kernel_nu,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            learn_kernel_beta=learn_kernel_beta,
            k_adaptation_strategy=k_adaptation_strategy,
            k_base=k_base,
            sampling_method=sampling_method,
            target_size=target_size,
            spatial_weight=spatial_weight,
        )
        
        # Extract image filename for response
        image_filename = os.path.basename(image_path) if image_path else None
        
        # Prepare results summary
        results_summary = {}
        for alg_name, result in results.items():
            if result['success']:
                results_summary[alg_name] = {
                    'success': True,
                    'runtime': result['runtime'],
                    'metrics': result['metrics']
                }
            else:
                results_summary[alg_name] = {
                    'success': False,
                    'error': result['error'],
                    'runtime': result['runtime']
                }
        
        return jsonify({
            'status': 'success',
            'results': results_summary,
            'dataset_info': {
                'n_cells': metadata['spatial_coords'].shape[0],
                'n_layers': len(set(metadata['cell_assignments'])),
                'spatial_shape': str(metadata['spatial_coords'].shape),
                'layer_counts': metadata['layer_counts']
            },
            'nystrom_enabled': nystrom_enabled,
            'image_path': f"results/{image_filename}" if image_filename else None,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in mouse brain cortical analysis: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


if __name__ == '__main__':
    app.run(debug=True) 