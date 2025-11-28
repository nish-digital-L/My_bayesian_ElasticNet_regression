from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み（日本語のパスに対応）
file_path = "your_file_name.csv"
data = pd.read_csv(file_path, encoding='utf-8')

# 列名を英数字に変換（記号をアンダースコアに）
data.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in data.columns]

# 特徴量（E列以降）とターゲット（KSS1y_IV）
features = data.columns[4:]
target = 'KSS1y_IV'

X = data[features]
y = data[target]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 欠損値補完
imputer = SimpleImputer(strategy="mean")
X_train_filled = imputer.fit_transform(X_train_scaled)
X_test_filled = imputer.transform(X_test_scaled)

# ElasticNetCVで変数選択
enet = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .9], random_state=42)
enet.fit(X_train_filled, y_train)
selected_features = np.where(enet.coef_ != 0)[0]
X_train_selected = X_train_filled[:, selected_features]

import pymc as pm
import arviz as az
if __name__ == '__main__':
    with pm.Model() as model:
        # ハイパーパラメータの事前分布（ラプラス分布に近づくように設計）
        tau = pm.HalfNormal('tau', sigma=1)
        
        # L1とL2の混合事前分布（ベイズElasticNet的な設計）
        beta = pm.Normal('beta', mu=0, sigma=tau, shape=X_train_selected.shape[1])
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # 線形予測モデル
        mu = intercept + pm.math.dot(X_train_selected, beta)

        # 尤度（正規分布）
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train)

        # MCMCによる事後分布のサンプリング
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # ArviZのsummaryで信用区間取得
    summary = az.summary(trace, var_names=['beta'], hdi_prob=0.95)

    # 変数名取得（インデックスから元データの列名に対応）
    feature_names = [features[i] for i in selected_features]  # 正確な対応付け

    # 整形してDataFrame化
    beta_data = pd.DataFrame({
        'Feature': [feature_names[int(idx.split('[')[1][:-1])] for idx in summary.index],
        'Mean': summary['mean'].values,
        'Lower': summary['hdi_2.5%'].values,
        'Upper': summary['hdi_97.5%'].values
    })

    # 有意性フラグを追加（ゼロをまたがないもの）
    beta_data['Significant'] = ~((beta_data['Lower'] <= 0) & (beta_data['Upper'] >= 0))

    # 絶対値のMeanを別列に保存
    beta_data['AbsMean'] = beta_data['Mean'].abs()

    # 有意性→係数の大きさ順に並び替え
    beta_data = beta_data.sort_values(by=['Significant', 'AbsMean'], ascending=[False, False])

    # エラーバー（非対称）
    lower_error = beta_data['Mean'] - beta_data['Lower']
    upper_error = beta_data['Upper'] - beta_data['Mean']
    error = [lower_error, upper_error]

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.barh(
        beta_data['Feature'],
        beta_data['Mean'],
        xerr=error,
        color='cornflowerblue',
        ecolor='gray',
        capsize=4,
        alpha=0.8
    )

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero reference')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Bayesian Elastic Net Coefficients with 95% Credible Intervals')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()

    for _, row in beta_data.iterrows():
        feature = row['Feature']
        mean_coef = row['Mean']
        hdi_lower = row['Lower']
        hdi_upper = row['Upper']
        
        print(f"{feature}: Coefficient = {mean_coef:.3f}, 95% CI = [{hdi_lower:.3f}, {hdi_upper:.3f}]")