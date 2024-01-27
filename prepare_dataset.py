import csv

class DatasetCreator:
    def __init__(self, label_map=None, output_file="dataset.csv"):
        """
        Initialize the DatasetCreator.

        :param label_map: A dictionary mapping labels to integers.
        :param output_file: Output CSV file name.
        """
        self.label_map = label_map or {}
        self.output_file = output_file
        self.data = []

    def add_data(self, text, label):
        """
        Add a data point to the dataset.

        :param text: Text data.
        :param label: Label associated with the text.
        """
        self.data.append((text, label))

    def create_dataset(self):
        """
        Create and save the dataset as a CSV file.
        """
        labeled_data = [(text, self.label_map.get(label, label)) for text, label in self.data]
        with open(self.output_file, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for text, label in labeled_data:
                writer.writerow([text, label])

if __name__ == "__main__":
    # Sample data - replace with your actual data
    data = [
        ("I love this product", "positive"),
        ("Horrible experience", "negative"),
        # Add more data points here...
    ]

    # Define label mapping if needed
    label_mapping = {"negative": 0, "positive": 1}

    # Create and save the dataset
    dataset_creator = DatasetCreator(label_map=label_mapping)
    for text, label in data:
        dataset_creator.add_data(text, label)
    dataset_creator.create_dataset()
