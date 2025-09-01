import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def feature_correlation_heatmap_english(csv_file_path):
    """
    Generate feature correlation heatmap in English
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read file: {csv_file_path}")
        print(f"Data shape: {df.shape}")
    except FileNotFoundError:
        print(f"File {csv_file_path} not found")
        return

    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Create a better column name mapping for display
    column_name_mapping = {
        'time_step': 'Time Step',
        'action': 'Action',
        'ac': 'AC',
        'heater': 'Heater',
        'dehum': 'Dehumidifier',
        'hum': 'Humidifier',
        'reward': 'Reward',
        'label_next': 'Next Label',
        'q_value_max': 'Max Q-value',
        'temp': 'Temperature',
        'humid': 'Humidity',
        'light': 'Light',
        'temp_normalized': 'Temp (Norm)',
        'humid_normalized': 'Humid (Norm)',
        'light_normalized': 'Light (Norm)'
    }

    # Calculate correlation matrix
    correlation = df[numeric_cols].corr()

    # Create figure
    plt.figure(figsize=(14, 12))

    # Create heatmap with better styling
    mask = np.triu(np.ones_like(correlation, dtype=bool))  # Mask upper triangle

    heatmap = sns.heatmap(
        correlation,
        annot=True,
        cmap='RdBu_r',
        center=0,
        fmt='.2f',
        mask=mask,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        annot_kws={"size": 9, "weight": "bold"},
        linewidths=0.5,
        linecolor='white'
    )

    # Improve tick labels
    readable_labels = [column_name_mapping.get(col, col.replace('_', ' ').title()) for col in correlation.columns]
    heatmap.set_xticklabels(readable_labels, rotation=45, ha='right', fontsize=10)
    heatmap.set_yticklabels(readable_labels, rotation=0, fontsize=10)

    plt.title('Feature Correlation Heatmap\n', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save the plot
    plt.savefig('correlation_heatmap_english.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print correlation insights
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS INSIGHTS")
    print("=" * 60)

    # Strong positive correlations (>0.7)
    strong_pos = []
    # Strong negative correlations (<-0.5)
    strong_neg = []

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            corr_val = correlation.iloc[i, j]
            if abs(corr_val) > 0.3:  # Only show meaningful correlations
                col1 = column_name_mapping.get(correlation.columns[i], correlation.columns[i])
                col2 = column_name_mapping.get(correlation.columns[j], correlation.columns[j])

                if corr_val > 0.7:
                    strong_pos.append((col1, col2, corr_val))
                elif corr_val < -0.5:
                    strong_neg.append((col1, col2, corr_val))

                print(f"{col1} ↔ {col2}: {corr_val:+.3f}")

    # Print strong correlations summary
    if strong_pos:
        print(f"\nSTRONG POSITIVE CORRELATIONS (>0.7):")
        for col1, col2, val in strong_pos:
            print(f"  {col1} ↔ {col2}: {val:+.3f}")

    if strong_neg:
        print(f"\nSTRONG NEGATIVE CORRELATIONS (<-0.5):")
        for col1, col2, val in strong_neg:
            print(f"  {col1} ↔ {col2}: {val:+.3f}")

    # Most correlated features with reward
    if 'reward' in correlation.columns:
        reward_corr = correlation['reward'].drop('reward')
        print(f"\nFEATURES MOST CORRELATED WITH REWARD:")
        for feature, corr_val in reward_corr.abs().sort_values(ascending=False).head(5).items():
            feature_name = column_name_mapping.get(feature, feature)
            actual_corr = correlation.loc['reward', feature]
            print(f"  {feature_name}: {actual_corr:+.3f}")


def detailed_correlation_analysis(csv_file_path):
    """
    More detailed correlation analysis with multiple heatmaps
    """
    df = pd.read_csv(csv_file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Create subplots for different correlation views
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    column_name_mapping = {
        'time_step': 'Time Step', 'action': 'Action', 'ac': 'AC', 'heater': 'Heater',
        'dehum': 'Dehumidifier', 'hum': 'Humidifier', 'reward': 'Reward',
        'label_next': 'Next Label', 'q_value_max': 'Max Q-value',
        'temp': 'Temperature', 'humid': 'Humidity', 'light': 'Light'
    }

    # 1. Full correlation matrix
    corr_matrix = df[numeric_cols].corr()
    readable_labels = [column_name_mapping.get(col, col) for col in corr_matrix.columns]

    im1 = axes[0, 0].matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('Full Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    axes[0, 0].set_xticks(range(len(corr_matrix.columns)))
    axes[0, 0].set_yticks(range(len(corr_matrix.columns)))
    axes[0, 0].set_xticklabels(readable_labels, rotation=45, ha='left', fontsize=9)
    axes[0, 0].set_yticklabels(readable_labels, fontsize=9)
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            axes[0, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center', fontsize=8,
                            fontweight='bold' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'normal')

    # 2. Reward-specific correlations
    if 'reward' in corr_matrix.columns:
        reward_corr = corr_matrix['reward'].drop('reward').sort_values()
        axes[0, 1].barh(range(len(reward_corr)), reward_corr.values,
                        color=['red' if x < 0 else 'green' for x in reward_corr.values])
        axes[0, 1].set_yticks(range(len(reward_corr)))
        axes[0, 1].set_yticklabels([column_name_mapping.get(col, col) for col in reward_corr.index], fontsize=9)
        axes[0, 1].set_title('Correlation with Reward', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel('Correlation Coefficient')

    # 3. Action-related correlations
    if 'action' in corr_matrix.columns:
        action_corr = corr_matrix['action'].drop('action').sort_values()
        axes[1, 0].barh(range(len(action_corr)), action_corr.values,
                        color=['red' if x < 0 else 'green' for x in action_corr.values])
        axes[1, 0].set_yticks(range(len(action_corr)))
        axes[1, 0].set_yticklabels([column_name_mapping.get(col, col) for col in action_corr.index], fontsize=9)
        axes[1, 0].set_title('Correlation with Action', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_xlabel('Correlation Coefficient')

    # 4. Environmental factors correlation
    env_cols = [col for col in numeric_cols if col in ['temp', 'humid', 'light', 'ac', 'heater', 'dehum', 'hum']]
    if len(env_cols) > 1:
        env_corr = df[env_cols].corr()
        im4 = axes[1, 1].matshow(env_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Environmental Factors Correlation', fontsize=14, fontweight='bold', pad=20)
        axes[1, 1].set_xticks(range(len(env_corr.columns)))
        axes[1, 1].set_yticks(range(len(env_corr.columns)))
        env_labels = [column_name_mapping.get(col, col) for col in env_corr.columns]
        axes[1, 1].set_xticklabels(env_labels, rotation=45, ha='left', fontsize=9)
        axes[1, 1].set_yticklabels(env_labels, fontsize=9)
        plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout()
    plt.savefig('detailed_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# Usage examples
if __name__ == "__main__":
    # Basic correlation heatmap
    feature_correlation_heatmap_english("rollout_results.csv")

    # For more detailed analysis (uncomment if needed)
    # detailed_correlation_analysis("rollout_results.csv")