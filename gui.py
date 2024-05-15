import tkinter
import tkinter.filedialog
import customtkinter as ctk
import torch
from finetuning import finetune

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Finetuned!")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure((1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Finetuned!", font=ctk.CTkFont(size=20, weight="bold", family="Comic Sans MS"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.menu_finetune = ctk.CTkButton(self.sidebar_frame)
        self.menu_finetune.grid(row=1, column=0, padx=20, pady=10)
        self.menu_models = ctk.CTkButton(self.sidebar_frame)
        self.menu_models.grid(row=2, column=0, padx=20, pady=10)
        self.menu_chat = ctk.CTkButton(self.sidebar_frame)
        self.menu_chat.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        self.console_toggle_button = ctk.CTkSwitch(self.sidebar_frame, text="Expert")
        self.console_toggle_button.grid(row=12, column=0, padx=10, pady=10)

        # create frame
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.main_frame.grid_columnconfigure(3, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=2)
        
        # create selector
        self.select_model = ctk.CTkLabel(self.main_frame, text="Select Model")
        self.select_model.grid(row=0, column=0, padx=10, pady=10)
        self.select_model_entry = ctk.CTkEntry(self.main_frame, placeholder_text="no model selected")
        self.select_model_entry.grid(row=0, column=1)
        self.select_model_button = ctk.CTkButton(self.main_frame, text="Browse", command=lambda: self.browse_directory(self.select_model_entry))
        self.select_model_button.grid(row=0, column=2, padx=10)
        
        self.select_dataset_label = ctk.CTkLabel(self.main_frame, text="Select Dataset:")
        self.select_dataset_label.grid(row=1, column=0, padx=10, pady=10)
        self.select_dataset_entry = ctk.CTkEntry(self.main_frame, placeholder_text="no dataset selected")
        self.select_dataset_entry.grid(row=1, column=1)
        self.select_dataset_button = ctk.CTkButton(self.main_frame, text="Browse", command=lambda: tkinter.filedialog.askopenfilename(self.select_dataset_entry))
        self.select_dataset_button.grid(row=1, column=2, padx=10)

        self.quantization_label = ctk.CTkLabel(self.main_frame, text="Quantization:")
        self.quantization_label.grid(row=2, column=0, padx=10, pady=10)
        self.quantization_var = ctk.StringVar(value="Q8")
        self.quantization_multiselect = ctk.CTkComboBox(self.main_frame, variable=self.quantization_var, values=["Q8", "Q4"])
        self.quantization_multiselect.grid(row=2, column=1)

        self.batch_size = ctk.CTkLabel(self.main_frame, text="Batch Size:")
        self.batch_size.grid(row=5, column=0, padx=10, pady=10)
        self.batch_size_entry = ctk.CTkEntry(self.main_frame)
        self.batch_size_entry.grid(row=5, column=1)

        self.epochs = ctk.CTkLabel(self.main_frame, text="Epochs:")
        self.epochs.grid(row=6, column=0, padx=10, pady=10)
        self.epochs_entry = ctk.CTkEntry(self.main_frame)
        self.epochs_entry.grid(row=6, column=1)

        self.lr_rate = ctk.CTkLabel(self.main_frame, text="Learning Rate:")
        self.lr_rate.grid(row=7, column=0, padx=10, pady=10)
        self.lr_rate_entry = ctk.CTkEntry(self.main_frame)
        self.lr_rate_entry.grid(row=7, column=1)

        check_gpu = "GPU available!" if torch.cuda.is_available() else "No GPU detected!"
        self.activate_gpu = ctk.CTkLabel(self.main_frame, text=check_gpu)
        self.activate_gpu.grid(row=8, column=0, padx=10, pady=10)

        self.start_btn = ctk.CTkButton(self.main_frame, text="Start", command=self.start_finetuning_script)
        self.start_btn.grid(row=9, column=0, padx=10, pady=10)

        # create charts frame
        self.charts_frame = ctk.CTkFrame(self)
        self.charts_frame.grid(row=0, rowspan=4, column=2, padx=(0, 20), pady=(20, 20), sticky="nsew")

        # set default values
        self.menu_finetune.configure(text="Finetune")
        self.menu_models.configure(state="disabled", text="Models")
        self.menu_chat.configure(state="disabled", text="Chat")
        self.batch_size_entry.insert(0, "32")
        self.epochs_entry.insert(0, "20")
        self.lr_rate_entry.insert(0, "0.001")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def browse_directory(self, entry_field):
        filepath = tkinter.filedialog.askdirectory()
        entry_field.delete(0, tkinter.END)
        entry_field.insert(0, filepath)

    def start_finetuning_script(self):
        finetune()

if __name__ == "__main__":
    app = App()
    app.mainloop()