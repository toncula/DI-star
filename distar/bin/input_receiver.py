import multiprocessing as mp
import tkinter as tk


def input_receiver(queue):
    def on_submit():
        new_input = entry.get()
        queue.put(new_input)
        entry.delete(0, tk.END)  # Clear the input field after submitting

    root = tk.Tk()
    root.title("Input Receiver")

    label = tk.Label(root, text="Enter new input:")
    label.pack()

    entry = tk.Entry(root, width=30)
    entry.pack()

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack()

    root.mainloop()


if __name__ == "__main__":
    queue = mp.Queue()
    input_process = mp.Process(target=input_receiver, args=(queue,))
    input_process.start()
    input_process.join()
