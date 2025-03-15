from DATA import DataProcessor

if __name__ == "__main__":
    url = "http://43.138.239.43:8000/get_daily_data/YAK4055565636/2299/20250121"
    data_processor = DataProcessor(url, "output.png")
    data_processor.plot_data()
