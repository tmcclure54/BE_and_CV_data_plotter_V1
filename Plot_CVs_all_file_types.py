import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ----- Data Parsing Helpers -----

def clean_column_name(name):
    #a suggestion of chatGPT
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def parse_data_file(file_path):
    #parse through a data file and pull out voltage and current values using generalizble attributes
    with open(file_path, 'r') as f:
        first_line = f.readline()
    if ',' in first_line:
        delimiter = ','
    elif '\t' in first_line:
        delimiter = '\t'
    else:
        delimiter = None

    try:
        df = pd.read_csv(file_path, delimiter=delimiter, engine='python')
    except Exception as e:
        raise ValueError(f"Failed to read file {file_path}: {e}")

    original_columns = list(df.columns)
    cleaned = [clean_column_name(str(c)) for c in original_columns]

    voltage_idx = None
    current_idx = None
    for i, name in enumerate(cleaned):
        if voltage_idx is None and ('potent' in name or 'volt' in name):
            voltage_idx = i
        if current_idx is None and 'curr' in name:
            current_idx = i

    if voltage_idx is None or current_idx is None:
        raise ValueError(f"Voltage/Current columns not found in {file_path}. Found: {original_columns}")

    voltage = df.iloc[:, voltage_idx]
    current = df.iloc[:, current_idx]
    return voltage, current


#Plotter
class CVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Multi-File Plotter")

        # controls for axes labels and title
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(control_frame, text="X-axis Label:").pack(side=tk.LEFT)
        self.x_label_var = tk.StringVar(value="Voltage (V)")
        tk.Entry(control_frame, textvariable=self.x_label_var, width=25).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, text="Y-axis Label:").pack(side=tk.LEFT)
        self.y_label_var = tk.StringVar(value="Current (mA) / Current Density (mA/cm²)")
        tk.Entry(control_frame, textvariable=self.y_label_var, width=30).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, text="Plot Title:").pack(side=tk.LEFT)
        self.title_var = tk.StringVar(value="Cyclic Voltammograms")
        tk.Entry(control_frame, textvariable=self.title_var, width=30).pack(side=tk.LEFT, padx=5)

        # Lists to store file paths and user inputs
        self.files = []
        self.legend_vars = []
        self.area_vars = []

        # Frame for file entries
        self.files_frame = tk.Frame(root)
        # Allow this frame to expand vertically
        self.files_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable container for file entries
        self.canvas = tk.Canvas(self.files_frame)
        self.scrollbar = tk.Scrollbar(self.files_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Buttons to manage files
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear Files", command=self.clear_files).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Plot CVs", command=self.plot_all).pack(side=tk.RIGHT, padx=5)

        # Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # add in scrolling and zooming functionality
        self.toolbar = NavigationToolbar2Tk(self.canvas_fig, root)
        self.toolbar.update()
        self.canvas_fig.get_tk_widget().pack()

        # asthetics for plotting
        sns.set_context("talk")
        sns.set_style("white")
        plt.rcParams.update({
            "font.family": "Arial",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.direction": 'in',
            "ytick.direction": 'in',
            "legend.fontsize": 12,
            "legend.title_fontsize": 14,
            "savefig.dpi": 300,
            "figure.dpi": 100
        })

        # Ensure window is sized to fit all widgets
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV & TXT files", "*.csv *.txt")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self._add_file_entry(p)

    def _add_file_entry(self, filepath):
        frame = tk.Frame(self.scrollable_frame, pady=2)
        frame.pack(fill=tk.X, expand=True)

        # File name label
        tk.Label(frame, text=os.path.basename(filepath), width=20, anchor="w").pack(side=tk.LEFT)

        # Legend title entry
        var_label = tk.StringVar(value=os.path.basename(filepath))
        tk.Label(frame, text="Label:").pack(side=tk.LEFT)
        tk.Entry(frame, textvariable=var_label, width=25).pack(side=tk.LEFT, padx=5)
        self.legend_vars.append(var_label)

        # Area entry
        var_area = tk.StringVar()
        tk.Label(frame, text="Area (cm²):").pack(side=tk.LEFT)
        tk.Entry(frame, textvariable=var_area, width=10).pack(side=tk.LEFT, padx=5)
        self.area_vars.append(var_area)

    def clear_files(self):
        self.files.clear()
        self.legend_vars.clear()
        self.area_vars.clear()
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.ax.clear()
        self.canvas_fig.draw()

    def plot_all(self):
        if not self.files:
            messagebox.showwarning("No files", "Please add at least one file.")
            return

        self.ax.clear()
        colors = sns.color_palette("husl", len(self.files))

        for idx, path in enumerate(self.files):
            try:
                voltage, current = parse_data_file(path)
                current_mA = current * -1000
                label_str = self.legend_vars[idx].get().strip() or os.path.basename(path)
                area_str = self.area_vars[idx].get().strip()

                if area_str:
                    try:
                        area = float(area_str)
                        data_y = current_mA / area
                        label_text = f"{label_str} (density)"
                    except ValueError:
                        messagebox.showwarning("Invalid Area",
                            f"Invalid area for {path}. Plotting raw current instead.")
                        data_y = current_mA
                        label_text = label_str
                else:
                    data_y = current_mA
                    label_text = label_str

                self.ax.plot(voltage, data_y, label=label_text,
                             color=colors[idx], linewidth=2)

            except Exception as e:
                messagebox.showerror("Parse Error", f"Error processing {path}:\n{e}")

        self.ax.set_xlabel(self.x_label_var.get())
        self.ax.set_ylabel(self.y_label_var.get())
        self.ax.set_title(self.title_var.get())
        self.ax.invert_xaxis()
        self.ax.legend(loc='best')
        self.fig.tight_layout()
        self.canvas_fig.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = CVPlotterApp(root)
    root.mainloop()
