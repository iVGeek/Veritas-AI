import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import time
import random
from datetime import datetime
import json
import os

class VeritasAILieDetector:
    """
    Veritas AI - Advanced Multi-Modal Lie Detection System
    Combines physiological, vocal, and behavioral analysis for deception detection
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Veritas AI - Advanced Lie Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Application state variables
        self.is_recording = False
        self.is_analyzing = False
        self.baseline_established = False
        self.current_session_data = []
        self.baseline_data = None
        
        # Machine learning model
        self.model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Create main interface
        self.setup_gui()
        self.load_or_train_model()
        
    def setup_gui(self):
        """Initialize the complete GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, text="Veritas AI - Advanced Lie Detection System", 
                              font=('Arial', 20, 'bold'), fg='#3498db', bg='#2c3e50')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Data visualization area
        self.setup_visualization_area(main_frame)
        
        # Results area
        self.setup_results_area(main_frame)
        
        # Question management
        self.setup_question_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Create the control panel with recording and analysis controls"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Baseline establishment
        baseline_btn = ttk.Button(control_frame, text="Establish Baseline", 
                                 command=self.establish_baseline)
        baseline_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Start recording
        self.record_btn = ttk.Button(control_frame, text="Start Recording", 
                                    command=self.toggle_recording)
        self.record_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Analyze data
        analyze_btn = ttk.Button(control_frame, text="Analyze Response", 
                                command=self.analyze_response)
        analyze_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Status indicators
        self.status_label = tk.Label(control_frame, text="Status: Ready", 
                                    bg='#2c3e50', fg='white')
        self.status_label.grid(row=1, column=0, columnspan=3, pady=5)
        
    def setup_visualization_area(self, parent):
        """Create the data visualization area"""
        viz_frame = ttk.LabelFrame(parent, text="Real-time Data Monitoring", padding="10")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Initialize plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.ax1.set_title('Heart Rate Variability', color='white')
        self.ax2.set_title('Voice Stress Analysis', color='white')
        self.ax3.set_title('Skin Conductance', color='white')
        self.ax4.set_title('Behavioral Indicators', color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_results_area(self, parent):
        """Create the results display area"""
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, 
                                                     bg='#34495e', fg='white', 
                                                     font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Confidence indicator
        self.confidence_frame = tk.Frame(results_frame, bg='#2c3e50')
        self.confidence_frame.pack(fill=tk.X, pady=5)
        
        self.confidence_label = tk.Label(self.confidence_frame, text="Confidence: ", 
                                        bg='#2c3e50', fg='white')
        self.confidence_label.pack(side=tk.LEFT)
        
        self.confidence_bar = ttk.Progressbar(self.confidence_frame, orient=tk.HORIZONTAL, 
                                             length=200, mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT, padx=5)
        
    def setup_question_panel(self, parent):
        """Create question management panel"""
        question_frame = ttk.LabelFrame(parent, text="Question Management", padding="10")
        question_frame.grid(row=1, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Question list
        tk.Label(question_frame, text="Question Bank:", bg='#2c3e50', fg='white').pack(anchor=tk.W)
        self.question_listbox = tk.Listbox(question_frame, height=8, bg='#34495e', fg='white')
        self.question_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Predefined questions
        self.default_questions = [
            "What is your full name?",
            "Where were you born?",
            "What is your current occupation?",
            "Did you complete the task as instructed?",
            "Are you telling the complete truth?",
            "Have you ever been convicted of a crime?",
            "Do you have any conflicts of interest?",
            "Is this your final answer?"
        ]
        
        for question in self.default_questions:
            self.question_listbox.insert(tk.END, question)
        
        # Question controls
        question_controls = tk.Frame(question_frame, bg='#2c3e50')
        question_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(question_controls, text="Add Question", 
                  command=self.add_question).pack(side=tk.LEFT, padx=2)
        ttk.Button(question_controls, text="Remove Question", 
                  command=self.remove_question).pack(side=tk.LEFT, padx=2)
        ttk.Button(question_controls, text="Ask Selected", 
                  command=self.ask_question).pack(side=tk.LEFT, padx=2)
        
        # Current question display
        self.current_question_label = tk.Label(question_frame, text="Current Question: None", 
                                              bg='#2c3e50', fg='#e74c3c', wraplength=300)
        self.current_question_label.pack(fill=tk.X, pady=5)
        
    def establish_baseline(self):
        """Establish physiological baseline for the subject"""
        self.status_label.config(text="Status: Establishing Baseline...")
        
        # Simulate baseline data collection
        threading.Thread(target=self.collect_baseline_data, daemon=True).start()
        
    def collect_baseline_data(self):
        """Collect baseline physiological data"""
        baseline_data = {
            'heart_rate': np.random.normal(70, 5, 100),  # Normal heart rate
            'voice_pitch': np.random.normal(120, 10, 100),  # Normal pitch
            'skin_conductance': np.random.normal(5, 1, 100),  # Normal conductance
            'response_time': np.random.normal(1.5, 0.3, 100)  # Normal response time
        }
        
        time.sleep(3)  # Simulate data collection time
        
        self.baseline_data = baseline_data
        self.baseline_established = True
        
        self.root.after(0, lambda: self.status_label.config(
            text="Status: Baseline Established Successfully"))
        
        self.update_plots(baseline_data, is_baseline=True)
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording physiological data"""
        if not self.baseline_established:
            messagebox.showwarning("Warning", "Please establish baseline first!")
            return
            
        self.is_recording = True
        self.record_btn.config(text="Stop Recording")
        self.status_label.config(text="Status: Recording...")
        
        # Start data collection thread
        threading.Thread(target=self.record_data, daemon=True).start()
        
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.status_label.config(text="Status: Recording Stopped")
        
    def record_data(self):
        """Simulate data recording from various sensors"""
        sample_count = 0
        max_samples = 200  # 20 seconds of data at 10Hz
        
        while self.is_recording and sample_count < max_samples:
            # Simulate sensor data collection
            current_data = {
                'timestamp': datetime.now(),
                'heart_rate': np.random.normal(75, 8) + random.choice([0, 5, 10]),  # Possible stress
                'voice_pitch': np.random.normal(125, 15) + random.choice([0, 8, 15]),
                'skin_conductance': np.random.normal(5.5, 1.5) + random.choice([0, 1, 2]),
                'response_time': np.random.normal(1.6, 0.4) + random.choice([0, 0.3, 0.6]),
                'micro_expressions': random.randint(0, 3),
                'eye_contact': random.uniform(0.6, 1.0)
            }
            
            self.current_session_data.append(current_data)
            sample_count += 1
            
            # Update plots every 10 samples
            if sample_count % 10 == 0:
                self.update_realtime_plots()
            
            time.sleep(0.1)  # 10Hz sampling rate
            
    def update_realtime_plots(self):
        """Update plots with real-time data"""
        if len(self.current_session_data) < 2:
            return
            
        # Extract recent data for plotting
        recent_data = self.current_session_data[-50:]  # Last 5 seconds
        
        plot_data = {
            'heart_rate': [d['heart_rate'] for d in recent_data],
            'voice_pitch': [d['voice_pitch'] for d in recent_data],
            'skin_conductance': [d['skin_conductance'] for d in recent_data],
            'timestamps': range(len(recent_data))
        }
        
        self.root.after(0, lambda: self.update_plots(plot_data, is_baseline=False))
        
    def update_plots(self, data, is_baseline=False):
        """Update all visualization plots"""
        color = 'green' if is_baseline else 'red'
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # Plot data
        if 'heart_rate' in data:
            self.ax1.plot(data['timestamps'], data['heart_rate'], color=color, linewidth=2)
            self.ax1.set_title('Heart Rate Variability', color='white')
            self.ax1.set_ylabel('BPM', color='white')
            
        if 'voice_pitch' in data:
            self.ax2.plot(data['timestamps'], data['voice_pitch'], color=color, linewidth=2)
            self.ax2.set_title('Voice Stress Analysis', color='white')
            self.ax2.set_ylabel('Pitch (Hz)', color='white')
            
        if 'skin_conductance' in data:
            self.ax3.plot(data['timestamps'], data['skin_conductance'], color=color, linewidth=2)
            self.ax3.set_title('Skin Conductance', color='white')
            self.ax3.set_ylabel('Conductance', color='white')
            
        # Behavioral indicators
        if not is_baseline and len(self.current_session_data) > 10:
            recent = self.current_session_data[-10:]
            micro_expr = np.mean([d['micro_expressions'] for d in recent])
            eye_contact = np.mean([d['eye_contact'] for d in recent])
            
            indicators = [micro_expr, 1 - eye_contact]  # Normalize
            self.ax4.bar(['Micro Expressions', 'Eye Contact Avoidance'], indicators, color=color)
            self.ax4.set_title('Behavioral Indicators', color='white')
            self.ax4.set_ylabel('Score', color='white')
        
        self.canvas.draw()
        
    def analyze_response(self):
        """Analyze the recorded response for deception indicators"""
        if len(self.current_session_data) < 10:
            messagebox.showwarning("Warning", "Not enough data recorded for analysis!")
            return
            
        self.status_label.config(text="Status: Analyzing Response...")
        threading.Thread(target=self.perform_analysis, daemon=True).start()
        
    def perform_analysis(self):
        """Perform comprehensive deception analysis"""
        # Extract features from recorded data
        features = self.extract_features(self.current_session_data)
        
        # Compare with baseline
        baseline_features = self.extract_features_baseline()
        
        # Calculate deviations
        deviations = self.calculate_deviations(features, baseline_features)
        
        # Machine learning prediction
        deception_probability = self.predict_deception(features)
        
        # Comprehensive analysis
        analysis_result = self.comprehensive_analysis(deviations, deception_probability)
        
        # Update UI with results
        self.root.after(0, lambda: self.display_results(analysis_result, deception_probability))
        
    def extract_features(self, data):
        """Extract relevant features from sensor data"""
        heart_rates = [d['heart_rate'] for d in data]
        voice_pitches = [d['voice_pitch'] for d in data]
        skin_conductance = [d['skin_conductance'] for d in data]
        response_times = [d['response_time'] for d in data]
        
        features = {
            'hr_mean': np.mean(heart_rates),
            'hr_std': np.std(heart_rates),
            'hr_variability': np.std(heart_rates) / np.mean(heart_rates),
            'voice_mean': np.mean(voice_pitches),
            'voice_std': np.std(voice_pitches),
            'conductance_mean': np.mean(skin_conductance),
            'conductance_std': np.std(skin_conductance),
            'response_mean': np.mean(response_times),
            'micro_expressions': np.mean([d['micro_expressions'] for d in data]),
            'eye_contact': np.mean([d['eye_contact'] for d in data])
        }
        
        return features
    
    def extract_features_baseline(self):
        """Extract features from baseline data"""
        if self.baseline_data is None:
            return None
            
        return self.extract_features([
            {
                'heart_rate': hr,
                'voice_pitch': vp,
                'skin_conductance': sc,
                'response_time': rt,
                'micro_expressions': 0,
                'eye_contact': 1.0
            }
            for hr, vp, sc, rt in zip(
                self.baseline_data['heart_rate'],
                self.baseline_data['voice_pitch'],
                self.baseline_data['skin_conductance'],
                self.baseline_data['response_time']
            )
        ])
    
    def calculate_deviations(self, features, baseline_features):
        """Calculate deviations from baseline"""
        if baseline_features is None:
            return {key: 0 for key in features.keys()}
            
        deviations = {}
        for key in features.keys():
            if baseline_features[key] != 0:
                deviations[key] = (features[key] - baseline_features[key]) / baseline_features[key]
            else:
                deviations[key] = 0
                
        return deviations
    
    def predict_deception(self, features):
        """Use ML model to predict deception probability"""
        if not self.model_trained:
            # Return simple heuristic based on features
            risk_factors = 0
            
            if features['hr_std'] > 10: risk_factors += 1
            if features['voice_std'] > 20: risk_factors += 1
            if features['conductance_mean'] > 7: risk_factors += 1
            if features['micro_expressions'] > 1.5: risk_factors += 1
            if features['eye_contact'] < 0.7: risk_factors += 1
            
            return risk_factors / 5 * 0.8  # Scale to 0-0.8 range
            
        # Use trained model if available
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        feature_vector = self.scaler.transform(feature_vector)
        return self.model.predict_proba(feature_vector)[0][1]
    
    def comprehensive_analysis(self, deviations, deception_probability):
        """Generate comprehensive analysis report"""
        analysis = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'deception_probability': deception_probability,
            'risk_factors': [],
            'confidence_indicators': [],
            'recommendations': []
        }
        
        # Analyze risk factors
        if deviations.get('hr_variability', 0) > 0.3:
            analysis['risk_factors'].append("Elevated heart rate variability detected")
        if deviations.get('voice_std', 0) > 0.4:
            analysis['risk_factors'].append("Voice stress patterns identified")
        if deviations.get('conductance_mean', 0) > 0.5:
            analysis['risk_factors'].append("Increased skin conductance suggests stress")
        
        # Confidence indicators
        if len(self.current_session_data) > 100:
            analysis['confidence_indicators'].append("Sufficient data collected for analysis")
        if self.baseline_established:
            analysis['confidence_indicators'].append("Reliable baseline established")
            
        # Recommendations
        if deception_probability < 0.3:
            analysis['recommendations'].append("Low deception probability - response appears truthful")
        elif deception_probability < 0.6:
            analysis['recommendations'].append("Moderate deception probability - recommend further questioning")
        else:
            analysis['recommendations'].append("High deception probability - significant indicators detected")
            
        return analysis
    
    def display_results(self, analysis, probability):
        """Display analysis results in the UI"""
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        result_text = f"=== VERITAS AI ANALYSIS REPORT ===\n"
        result_text += f"Timestamp: {analysis['timestamp']}\n"
        result_text += f"Deception Probability: {probability:.2%}\n\n"
        
        result_text += "RISK FACTORS:\n"
        for factor in analysis['risk_factors']:
            result_text += f"• {factor}\n"
            
        result_text += "\nCONFIDENCE INDICATORS:\n"
        for indicator in analysis['confidence_indicators']:
            result_text += f"• {indicator}\n"
            
        result_text += "\nRECOMMENDATIONS:\n"
        for recommendation in analysis['recommendations']:
            result_text += f"• {recommendation}\n"
            
        self.results_text.insert(1.0, result_text)
        
        # Update confidence bar
        self.confidence_bar['value'] = probability * 100
        self.confidence_label.config(text=f"Confidence: {probability:.2%}")
        
        self.status_label.config(text="Status: Analysis Complete")
        
    def add_question(self):
        """Add a new question to the question bank"""
        new_question = tk.simpledialog.askstring("Add Question", "Enter new question:")
        if new_question:
            self.question_listbox.insert(tk.END, new_question)
            
    def remove_question(self):
        """Remove selected question from the question bank"""
        selection = self.question_listbox.curselection()
        if selection:
            self.question_listbox.delete(selection[0])
            
    def ask_question(self):
        """Display the selected question"""
        selection = self.question_listbox.curselection()
        if selection:
            question = self.question_listbox.get(selection[0])
            self.current_question_label.config(text=f"Current Question: {question}")
            
            # Log the question asking
            self.results_text.insert(tk.END, f"\n[Q] {datetime.now().strftime('%H:%M:%S')}: {question}\n")
            
    def load_or_train_model(self):
        """Load pre-trained model or train a new one"""
        model_path = "lie_detector_model.pkl"
        scaler_path = "scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.model_trained = True
                print("Pre-trained model loaded successfully")
            except:
                self.train_model()
        else:
            self.train_model()
            
    def train_model(self):
        """Train a machine learning model for lie detection"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [hr_mean, hr_std, hr_variability, voice_mean, voice_std, 
        #           conductance_mean, conductance_std, response_mean, micro_expressions, eye_contact]
        X_truthful = np.random.normal([70, 5, 0.07, 120, 10, 5, 1, 1.5, 0.5, 0.9], 
                                    [5, 2, 0.02, 8, 3, 1, 0.3, 0.2, 0.3, 0.1], 
                                    (n_samples//2, 10))
        
        X_deceptive = np.random.normal([80, 10, 0.12, 135, 20, 7, 2, 2.0, 2.0, 0.7], 
                                     [8, 4, 0.04, 15, 6, 2, 0.8, 0.5, 0.8, 0.2], 
                                     (n_samples//2, 10))
        
        X = np.vstack([X_truthful, X_deceptive])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        with open("lie_detector_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open("scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        self.model_trained = True
        print("Model trained and saved successfully")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = VeritasAILieDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()


#### add more features 