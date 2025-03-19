import requests
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def remove_overlapping(df):
    """
    Remove overlapping intervals based on citta_mean values.
    
    Args:
        df (pd.DataFrame):
            the DataFrame containing citta_mean, motion_mean, and heartrate_mean columns

    Returns:
        pd.DataFrame, the DataFrame with overlapping intervals removed

    """
    intervals = []
    for label in df.index:
        try:
            start_str, end_str = label.split(" ~ ")
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
        except Exception:
            start, end = pd.NaT, pd.NaT
        intervals.append((label, start, end, df.loc[label, 'citta_mean']))
    
    # 按citta均值从高到低排序
    sorted_intervals = sorted(intervals, key=lambda x: x[3], reverse=True)
    
    selected = []
    for cur_label, cur_start, cur_end, cur_citta in sorted_intervals:
        overlap = False
        for sel_label, sel_start, sel_end, sel_citta in selected:
            # 判断区间是否重叠：若当前区间与已选择区间不满足互不重叠条件，则为重叠
            if not (cur_end <= sel_start or cur_start >= sel_end):
                overlap = True
                break
        if not overlap:
            selected.append((cur_label, cur_start, cur_end, cur_citta))
    
    # 返回按起始时间顺序排序的结果
    selected_labels = [item[0] for item in sorted(selected, key=lambda x: x[1])]
    
    return df.loc[selected_labels]
    
class DataProcessor:
    """
    A class to process data from a given URL.
    including getting data, processing data, dividing groups, and plotting data.

    Attributes:
        url (str): 
            the URL to get data
        data_json (dict):
            the data in JSON format
        df_result (pd.DataFrame):
            the DataFrame with sorted data
        group_list (list):
            the list of DataFrames for each group
        group_min_max_records (dict):
            the dictionary containing the record with the highest and lowest citta_mean values for each group
        group_data_mean_records (dict):
            the dictionary containing the mean of citta_mean and heartrate_mean for each group
        image_path (str):
            the path to save the image
    """
    def __init__(self, url=None, image_path="output.png"):
        self.url = url
        self.data_json = None
        self.df_result = None
        self.group_list = None
        self.group_min_max_records = None
        self.group_data_mean_records = None
        self.image_path = image_path
        if url:
            self.process_data(url)

    def get_raw_data(self, url = None):
        """
        Get data from the given URL.
        
        Args:
            url (str): 
                the URL to get data

        Raises:
            requests.exceptions.RequestException: if the request fails
        """
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # 检查请求是否成功
            print("请求成功")
        except requests.exceptions.RequestException as e:
            print("请求失败:", e)
            return None

        self.data_json = json.loads(response.text)



    def process_data_15min(self, url=None):
        """
        process data from the given URL.
        create data windows for every 15 minutes and calculate the mean of citta, motion, and heartrate.
        sort the data based on citta_mean and motion_mean.
        remove the data with discontinuity.

        Args:
            url (str): 
                the URL to get data
        """
        if self.data_json.get("status") == "error":
            sys.exit("请求错误: {}".format(self.data_json.get("message")))
        res = self.data_json['data']
        data_dict = {
            "timestamp": list(res.keys()),
            "citta": [d.get("gcyy") for d in res.values()],
            "heartrate": [d.get("HR") for d in res.values()],
            "motion": [d.get("motion") for d in res.values()]
        }

        # 构造DataFrame
        data = pd.DataFrame(data_dict)
        data.set_index('timestamp', inplace=True)

        # 删除行索引带.1后缀的行，即无用数据
        data = data[~data.index.astype(str).str.endswith('.1')]
        data.index = pd.to_datetime(data.index)

        # 根据已有数据的最小和最大时间生成3分钟时间序列，并填充缺失数据为0
        full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='3min')
        data = data.reindex(full_index, fill_value=0)

        data.index = pd.to_datetime(data.index)
        # 分别计算citta、motion和heartrate的滚动均值（窗口18min，对应约15分钟均值，跳过前五个点以修正对齐）
        citta_mean_15min = data['citta'].rolling('18min').mean().iloc[5:]
        motion_mean_15min = data['motion'].rolling('18min').mean().iloc[5:]
        heartrate_mean_15min = data['heartrate'].rolling('18min').mean().iloc[5:]
        

        initial_time = data.index[0]
    
        window_length = pd.Timedelta(minutes=15)
        current_start = initial_time
            
        # 建立组别数据：每组包含时间标签、citta均值、motion均值、heartrate均值及各自不连续标记
        group_list = []
        num_windows = len(citta_mean_15min)
        for i in range(num_windows):
            current_end = current_start + window_length
            label = "{} ~ {}".format(current_start.strftime("%Y-%m-%d %H:%M:%S"),
                                    current_end.strftime("%Y-%m-%d %H:%M:%S"))
            window_data = data.loc[current_start: current_end]
            mark_citta = " [include discontinuity]" if (window_data['citta'] == 0).any() else ""
            mark_motion = " [include discontinuity]" if (window_data['motion'] == 0).any() else ""
            mark_heartrate = " [include discontinuity]" if (window_data['heartrate'] == 0).any() else ""
                
            # 从之前计算的滚动均值Series中取出对应值
            citta_avg = citta_mean_15min.iloc[i]
            motion_avg = motion_mean_15min.iloc[i]
            heartrate_avg = heartrate_mean_15min.iloc[i]
        
            # 将当前组别数据加入列表
            group_list.append((label, citta_avg, motion_avg, heartrate_avg, mark_citta, mark_motion, mark_heartrate))
            current_start += pd.Timedelta(minutes=3)
            
        # 按citta均值为主关键字，其次按motion均值从高到低排序
        sorted_groups = sorted(group_list,
                            key=lambda x: ((x[1] if pd.notnull(x[1]) else -np.inf),
                                            (x[2] if pd.notnull(x[2]) else -np.inf)),
                            reverse=True)
            
        # 删除包含 discontinuity 标记的组别（只要citta、motion或heartrate中任一项包含标记即删除）
        filtered_groups = [group for group in sorted_groups if "include discontinuity" not in (group[4] + group[5] + group[6])]
            
        # 重新构造DataFrame，排序时仍以citta均值为主关键字，motion均值为次要关键字
        df_result = pd.DataFrame(
            {'citta_mean': [item[1] for item in filtered_groups],
            'motion_mean': [item[2] for item in filtered_groups],
            'heartrate_mean': [item[3] for item in filtered_groups]},
            index=[item[0] for item in filtered_groups]
        )

        self.df_result = df_result



    def divide_groups(self, df_result):
        """
        divide the data into 6 groups based on motion_mean values.
        for each group, find the record with the highest and lowest citta_mean values.
        calculate the mean of citta_mean, heartrate_mean and motion_mean for each group.

        Args:
            df_result (pd.DataFrame):
                the DataFrame containing citta_mean, motion_mean, and heartrate_mean columns
        """
        # 使用筛选条件构造各组，并对各组按照citta_mean从大到小排序
        df_group1 = remove_overlapping(df_result[(df_result['motion_mean'] >= 0) & (df_result['motion_mean'] < 2.5)]).sort_values('citta_mean', ascending=False)
        df_group2 = remove_overlapping(df_result[(df_result['motion_mean'] >= 2.5) & (df_result['motion_mean'] < 5)]).sort_values('citta_mean', ascending=False)
        df_group3 = remove_overlapping(df_result[(df_result['motion_mean'] >= 5) & (df_result['motion_mean'] < 25)]).sort_values('citta_mean', ascending=False)
        df_group4 = remove_overlapping(df_result[(df_result['motion_mean'] >= 25) & (df_result['motion_mean'] < 60)]).sort_values('citta_mean', ascending=False)
        df_group5 = remove_overlapping(df_result[(df_result['motion_mean'] >= 60) & (df_result['motion_mean'] < 80)]).sort_values('citta_mean', ascending=False)
        df_group6 = remove_overlapping(df_result[df_result['motion_mean'] >= 80]).sort_values('citta_mean', ascending=False)

        group_list = [df_group1, df_group2, df_group3, df_group4, df_group5, df_group6]
        # 记录每组中最高和最低的记录（基于 citta_mean 值），并用变量保存
        group_names = [("calm1", df_group1), ("clam2", df_group2), ("sober", df_group3),
                    ("moving", df_group4), ("exercise", df_group5), ("exercise plus", df_group6)]
        group_min_max_records = {}
        group_data_mean_records = {}
        for group_name, group_df in group_names:
            if not group_df.empty:
                max_label = group_df['citta_mean'].idxmax()
                max_record = group_df.loc[max_label]
                min_label = group_df['citta_mean'].idxmin()
                min_record = group_df.loc[min_label]
                group_min_max_records[group_name] = {"max": (max_label, max_record), "min": (min_label, min_record)}
                group_data_mean_records[group_name] = {"citta_mean": group_df['citta_mean'].mean(),
                                                    "heartrate_mean": group_df['heartrate_mean'].mean(),
                                                    "motion_mean": group_df['motion_mean'].mean()}
            else:
                group_min_max_records[group_name] = None

        self.group_list = group_list
        self.group_min_max_records = group_min_max_records
        self.group_data_mean_records = group_data_mean_records



    def process_data(self, url=None):
        """
        process data from the given URL.
        create data windows for every 15 minutes and calculate the mean of citta, motion, and heartrate.
        sort the data based on citta_mean and motion_mean.
        remove the data with discontinuity.
        divide the data into 6 groups based on motion_mean values.
        for each group, find the record with the highest and lowest citta_mean values.
        calculate the mean of citta_mean, heartrate_mean and motion_mean for each group.

        Args:
            url (str): 
                the URL to get data
        """
        if url:
            self.url = url
            self.get_raw_data(url)
            self.process_data_15min(url)
            self.divide_groups(self.df_result)
        else:
            print("URL is not provided.")

       
    
    def get_processed_data(self):
        """
        get the processed data.

        Returns:
            str, the JSON format of the processed data
        """

        # 将 DataFrame 和字典数据转换为可序列化的 JSON 格式
        output = {}
        output["df_result"] = self.df_result.to_dict(orient="index")
        output["group_list"] = [group.to_dict(orient="index") for group in self.group_list]
        
        group_min_max_serialized = {}
        for group_name, records in self.group_min_max_records.items():
            if records is None:
                group_min_max_serialized[group_name] = None
            else:
                max_label, max_record = records["max"]
                min_label, min_record = records["min"]
                group_min_max_serialized[group_name] = {
                    "max": {max_label: max_record.to_dict()},
                    "min": {min_label: min_record.to_dict()}
                }
        output["group_min_max_records"] = group_min_max_serialized
        output["group_data_mean_records"] = self.group_data_mean_records
        
        return json.dumps(output)

    def plot_data(self, image_path=None):
        """
        plot the citta_mean and heartrate_mean over time.
        the motion_mean values are used to divide the data into different intervals.
        the color of the line represents the motion_mean values.
        the data points with the highest and lowest citta_mean values are annotated on the plot.
        the image is saved to the given path.

        Args:
            image_path (str):
                the path to save the image

        Returns:
            None   
        """
        if image_path is None:
            image_path = self.image_path
        df_result = self.df_result
        group_min_max_records = self.group_min_max_records

        df_result_sorted = df_result.copy()
        df_result_sorted['start_dt'] = pd.to_datetime(df_result_sorted.index.str.split(" ~ ").str[0])
        df_result_sorted.sort_values('start_dt', inplace=True)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        # 绘制 citta_mean 曲线，并加入图例
        ax1.plot(df_result_sorted['start_dt'], df_result_sorted['citta_mean'], color='grey',
                marker='o', markersize=3, linestyle='-', label="Citta Mean")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Citta Mean", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("Citta Mean and Heartrate Mean over Time")

        # 创建一个共享 x 轴的第二纵轴来绘制 heartrate_mean
        ax2 = ax1.twinx()

        # 根据 motion_mean 分区间，定义分区和对应颜色
        bins = [0, 2.5, 5, 25, 60, 80, np.inf]
        colors = ['#4B0082', 'plum', 'navy', 'darkgreen', 'orange', 'red']
        labels_bin = ["[0, 2.5)", "[2.5, 5)", "[5, 25)", "[25, 60)", "[60, 80)", "[80, ∞)"]

        df_result_sorted['motion_bin'] = pd.cut(df_result_sorted['motion_mean'],
                                                bins=bins,
                                                right=False,
                                                labels=labels_bin)

        # 按时间顺序将不同颜色的区间连起来
        segments = []
        seg_start = 0
        motion_bins_array = df_result_sorted['motion_bin'].values
        for i in range(1, len(motion_bins_array)):
            if motion_bins_array[i] != motion_bins_array[i-1]:
                segments.append(df_result_sorted.iloc[seg_start:i])
                seg_start = i
        segments.append(df_result_sorted.iloc[seg_start:])

        # 绘制 heartrate_mean 曲线，第一次出现时添加图例，后续不添加
        legend_added = set()
        for seg in segments:
            bin_label = seg['motion_bin'].iloc[0]
            if pd.isna(bin_label):
                continue
            # 根据 labels_bin 找到对应的颜色
            color = None
            for lbl, c in zip(labels_bin, colors):
                if lbl == bin_label:
                    color = c
                    break
            if color is None:
                continue
            if bin_label not in legend_added:
                ax2.plot(seg['start_dt'], seg['heartrate_mean'],
                        color=color, marker='o', markersize=4, linestyle='-', label=f"motion {bin_label}")
                legend_added.add(bin_label)
            else:
                ax2.plot(seg['start_dt'], seg['heartrate_mean'],
                        color=color, marker='o', markersize=4, linestyle='-')

        # 连接不同区间：用前一区间的颜色连接当前区间的起点和前一区间的终点
        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            cur_seg = segments[i]
            prev_bin = prev_seg['motion_bin'].iloc[0]
            color = None
            for lbl, c in zip(labels_bin, colors):
                if lbl == prev_bin:
                    color = c
                    break
            if color is None:
                continue
            ax2.plot([prev_seg['start_dt'].iloc[-1], cur_seg['start_dt'].iloc[0]],
                    [prev_seg['heartrate_mean'].iloc[-1], cur_seg['heartrate_mean'].iloc[0]],
                    color=color, linestyle='-', marker='o', markersize=4)

        ax2.set_ylabel("Heartrate Mean", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 获取两个坐标轴的图例句柄和标签
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        # 按 motion 区间从小到大排序图例：
        sorted_motion = []
        for handle, label in zip(h2, l2):
            if label and label.startswith("motion "):
                # 获取 motion_bin 标签
                mb = label[7:]
                if mb in labels_bin:
                    sorted_motion.append((labels_bin.index(mb), handle, label))
        sorted_motion.sort(key=lambda x: x[0])
        sorted_motion_handles = [item[1] for item in sorted_motion]
        sorted_motion_labels = [item[2] for item in sorted_motion]

        # 合并 citta_mean 图例和排序后的 motion 图例
        combined_handles = h1 + sorted_motion_handles
        combined_labels = l1 + sorted_motion_labels

        plt.legend(combined_handles, combined_labels, loc='best')
        
        # 定义不同组别的颜色映射
        group_color = {
            "calm1": "#4B0082",
            "clam2": "plum",
            "sober": "navy",
            "moving": "darkgreen",
            "exercise": "orange",
            "exercise plus": "red"
        }
        
        # === 以下新增代码：在 citta_mean 曲线上标出各组记录点对应的 citta_mean 与 heartrate_mean 值，若为同一点只标注一次 ===
        annotations = []  # 用于保存所有标注点的信息
        annotated_points = set()  # 用来记录已标注的点，key 为 (x, y)

        def annotate_point(label_str, record, color):
            start_time_str = label_str.split(" ~ ")[0]
            x_val = pd.to_datetime(start_time_str)
            y_val = record['citta_mean']
            # 使用 (x_val, round(y_val, 4)) 作为唯一标识
            point_key = (x_val, round(y_val, 4))
            if point_key in annotated_points:
                return None
            annotated_points.add(point_key)
            text = f"{record['citta_mean']:.2f}\n{record['heartrate_mean']:.2f}"
            annot = ax1.annotate(text, (x_val, y_val),
                                xytext=(10, -10), textcoords="offset points",
                                ha='left', va='top', fontsize=10, color=color,
                                arrowprops=dict(arrowstyle="-", color=color))
            return {'time': x_val, 'citta': y_val,
                    'text': text, 'annotation': annot,
                    'offset': (10, -10), 'extended': False,
                    'color': color}

        for group_name, records in group_min_max_records.items():
            ann_color = group_color.get(group_name, "blue")
            if records is None:
                continue
            # 判断单组内最高与最低是否为同一点
            if records["max"][0] == records["min"][0]:
                item = annotate_point(records["max"][0], records["max"][1], ann_color)
                if item:
                    annotations.append(item)
            else:
                for rec_type in ["max", "min"]:
                    item = annotate_point(records[rec_type][0], records[rec_type][1], ann_color)
                    if item:
                        annotations.append(item)

        # 对于任意两点，若二者的 citta_mean 差值在 0.15 以内且时间差在 30 分钟以内，
        # 则加长其中一个点（这里选择第一个未被加长的点）的直线（通过增加 xytext offset 实现）
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                a = annotations[i]
                b = annotations[j]
                time_diff = abs((a['time'] - b['time']).total_seconds()) / 60.0
                if abs(a['citta'] - b['citta']) <= 0.15 and time_diff <= 30:
                    if not a['extended']:
                        text_to_use = a['text']
                        base_coords = (a['time'], a['citta'])
                        a['annotation'].remove()
                        new_offset = (30, -30)
                        new_annot = ax1.annotate(text_to_use, base_coords,
                                                xytext=new_offset, textcoords="offset points",
                                                ha='left', va='top', fontsize=10, color=a['color'],
                                                arrowprops=dict(arrowstyle="-", color=a['color']))
                        a.update({'annotation': new_annot, 'offset': new_offset, 'extended': True})
        
        plt.savefig(image_path)

