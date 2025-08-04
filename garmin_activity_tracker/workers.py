from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd

class LoadDataWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # df_summary, df_running, df_running_splits
    enable_sync_button = pyqtSignal()

    def __init__(self, tracker, use_api=False, config=None):
        super().__init__()
        self.tracker = tracker
        self.use_api = use_api
        self.config = config

    def run(self):
        try:
            self.tracker.load_column_map()
            if not self.use_api:
                self.progress.emit("Loading local data without API...")
            else:
                self.progress.emit("Loading data from Garmin API...")

            max_activities = self.config.get("api", {}).get("max_activities")
            batch_size = self.config.get("api", {}).get("batch_size")
            df_summary = self.tracker.sync_summary_data(max_activities, batch_size, self.use_api)
        except Exception as e:
            self.progress.emit(f"Error during sync summary or with signal_map.json: {e}")
            self.finished.emit(None)
            return

        try:
            if not isinstance(df_summary, pd.DataFrame) or df_summary.empty:
                self.progress.emit("No activities found. Please sync with Garmin API.")
                self.finished.emit(None)
                self.enable_sync_button.emit()
                return

            self.progress.emit("Preprocessing local running data...")
            df_running = self.tracker.preprocess_running_data(df_summary.copy(), self.config)
            df_cycling = self.tracker.preprocess_cycling_data(df_summary.copy(), self.config)
            df_swimming = []
            df_workouts = self.tracker.preprocess_strength_training_data(df_summary.copy(), self.config)
        except Exception as e:
            self.progress.emit(f"Error during preprocessing: {e}")
            self.finished.emit(None)
            return

        try:
            self.progress.emit("Syncing local split data...")
            df_running_splits = self.tracker.sync_split_data(df_running, use_api=self.use_api)
            self.progress.emit("Data load complete.")

                        # Build the activities dictionary
            activities = {
                'All': df_summary,
                'Running': {
                    'Summary': df_running,
                    'Splits': df_running_splits,
                    'TSS': None  # Placeholder for TSS, calculated later
                },
                'Cycling': {
                    'Summary': df_cycling,
                    'Splits': None,  # Placeholder for cycling splits
                    'TSS': None  # Placeholder for TSS, calculated later
                },
                'Swimming': {
                    'Summary': df_swimming,
                    'Splits': None,
                    'TSS': None
                },
                'Workouts': {
                    'Summary': df_workouts,
                    'Splits': None,
                    'TSS': None
                }
            }

            self.progress.emit("Data load complete.")
            self.progress.emit("Press Sync Latest Data to download new data.")
            self.finished.emit(activities)
        except Exception as e:
            self.progress.emit(f"Error during data load: {e}")
            self.finished.emit(None)

class AIWorker(QThread):
    chunk_ready = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, prompt_content, model=None):
        super().__init__()
        self.prompt_content = prompt_content
        self.model = model

    def run(self):
        try:
            from ollama import chat  # Import here to avoid issues if not installed

            messages = [{"role": "user", "content": self.prompt_content}]
            ai_response = ""
            for chunk in chat(model=self.model, messages=messages, stream=True):
                text = chunk["message"]["content"]
                ai_response += text
                self.chunk_ready.emit(text)
            self.finished.emit(ai_response)
        except Exception as e:
            self.finished.emit(f"<b>Error:</b> {e}")