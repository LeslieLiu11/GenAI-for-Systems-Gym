import pandas as pd
import matplotlib.pyplot as plt

def plot_normalized_hit_rate(csv_file, output_png="normalized_hit_rate.png"):
    """
    CSV ：
        model,raw_hit_rate
        mlp_width_32,0.26
        mlp_width_64,0.29
        ...
    """
    # 讀取 CSV
    df = pd.read_csv(csv_file)

    max_rate = df['raw_hit_rate'].max()
    
    df['normalized_hit_rate'] = df['raw_hit_rate'] / max_rate
 
    
    # 繪圖
    plt.figure(figsize=(8,6))
    plt.bar(df['model'], df['normalized_hit_rate'], color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("Normalized Hit Rate")
    plt.title("Comparison of Normalized Hit Rates Across Models")
    plt.ylim([0, 1.05])  # 讓 y 軸最高超過 1 一點
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 儲存圖檔
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":

    csv_file = "eval.csv"
    plot_normalized_hit_rate(csv_file, "normalized_hit_rate.png")
