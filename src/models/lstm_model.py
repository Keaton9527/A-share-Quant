"""
LSTM模型 - 用于预测股价涨跌方向的LSTM模型（PyTorch实现）
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel


class LSTMNet(nn.Module):
    """PyTorch LSTM网络定义"""
    
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2, output_dim=1):
        """
        初始化LSTM网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout概率
            output_dim: 输出维度
        """
        super(LSTMNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据，形状为(batch_size, seq_len, input_dim)
            
        Returns:
            输出，形状为(batch_size, output_dim)
        """
        # LSTM输出
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # Dropout
        last_out = self.dropout(last_out)
        
        # 全连接层
        fc_out = self.fc(last_out)
        
        # Sigmoid激活
        out = self.sigmoid(fc_out)
        
        return out


class LSTMModel(BaseModel):
    """LSTM模型类，用于预测股价涨跌方向"""
    
    def __init__(self, sequence_length=10, hidden_dim=50, num_layers=2, name="LSTM"):
        """
        初始化LSTM模型
        
        Args:
            sequence_length: 输入序列长度（使用多少天的历史数据）
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            name: 模型名称
        """
        super().__init__(name)
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_scaler = MinMaxScaler()  # 特征缩放器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def build_model(self, input_dim):
        """
        构建LSTM模型
        
        Args:
            input_dim: 输入特征维度
            
        Returns:
            构建的模型
        """
        model = LSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        return model
    
    def preprocess(self, df):
        """
        数据预处理，提取特征并标准化
        
        Args:
            df: 包含股票数据的DataFrame
            
        Returns:
            处理后的特征数据和标签
        """
        # 确保数据是按日期升序排列的
        df_sorted = df.sort_values('trade_date', ascending=True)
        
        # 提取特征
        features = []
        
        # 价格特征
        if 'close_pre_adj' in df_sorted.columns:
            features.extend(['open_pre_adj', 'high_pre_adj', 'low_pre_adj', 'close_pre_adj'])
        else:
            features.extend(['open', 'high', 'low', 'close'])
        
        # 交易量特征
        if 'vol' in df_sorted.columns:
            features.append('vol')
        
        # 技术指标特征（如果有）
        tech_features = ['K', 'D', 'J', 'DIF', 'DEA', 'MACD']
        for feature in tech_features:
            if feature in df_sorted.columns:
                features.append(feature)
        
        # 提取特征数据
        data = df_sorted[features].values
        
        # 特征标准化
        if not hasattr(self, 'feature_scaler_fitted') or not self.feature_scaler_fitted:
            self.feature_scaler.fit(data)
            self.feature_scaler_fitted = True
        
        scaled_data = self.feature_scaler.transform(data)
        
        # 创建标签：1表示上涨，0表示下跌或持平
        close_col = 'close_pre_adj' if 'close_pre_adj' in df_sorted.columns else 'close'
        df_sorted['return'] = df_sorted[close_col].pct_change()
        df_sorted['label'] = (df_sorted['return'] > 0).astype(int)
        
        # 数据集X和y
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i+self.sequence_length])
            y.append(df_sorted['label'].iloc[i+self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, **kwargs):
        """
        训练LSTM模型
        
        Args:
            X: 特征数据，形状为(样本数, 序列长度, 特征数)
            y: 标签数据，形状为(样本数,)
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批量大小
            **kwargs: 其他训练参数
            
        Returns:
            训练历史
        """
        # 确定输入特征维度
        _, seq_len, n_features = X.shape
        
        # 构建模型
        if self.model is None:
            self.model = self.build_model(input_dim=n_features)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # 划分训练集和验证集
        val_size = int(len(X) * validation_split)
        train_size = len(X) - val_size
        
        X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
        X_val, y_val = X_tensor[train_size:], y_tensor[train_size:]
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 早停设置
        patience = 10
        best_val_loss = float('inf')
        counter = 0
        best_model_state = None
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            
            # 训练批次循环
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累加损失
                train_loss += loss.item()
                
                # 计算准确率
                predicted = (outputs >= 0.5).float()
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
            
            # 计算平均训练损失和准确率
            train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train
            
            # 验证模式
            self.model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 累加损失
                    val_loss += loss.item()
                    
                    # 计算准确率
                    predicted = (outputs >= 0.5).float()
                    total_val += batch_y.size(0)
                    correct_val += (predicted == batch_y).sum().item()
            
            # 计算平均验证损失和准确率
            val_loss = val_loss / len(val_loader)
            val_acc = correct_val / total_val
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # 打印进度
            if epochs <= 10 or epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停: 验证损失在{patience}个epoch内没有改善")
                    break
        
        # 恢复最佳模型权重
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        预测涨跌方向
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果，1表示上涨，0表示下跌
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测模式
        self.model.eval()
        
        with torch.no_grad():
            # 获取概率预测
            outputs = self.model(X_tensor)
            
            # 转换为二元分类结果
            predictions = (outputs >= 0.5).float().cpu().numpy()
        
        return predictions
    
    def predict_for_stock(self, df):
        """
        为单只股票预测未来一天的涨跌方向
        
        Args:
            df: 包含股票数据的DataFrame
            
        Returns:
            预测结果：1(买入)、0(持有)、-1(卖出)
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
        
        # 确保数据是按日期升序排列的
        df_sorted = df.sort_values('trade_date', ascending=True)
        
        # 提取并预处理最近的数据序列
        X, _ = self.preprocess(df_sorted)
        
        if len(X) == 0:
            print("警告：数据量不足，无法预测")
            return 0
        
        # 获取最后一个序列
        latest_sequence = X[-1:]
        
        # 预测
        prediction = self.predict(latest_sequence)[0][0]
        
        # 转换为交易信号：1(买入)、-1(卖出)
        if prediction == 1:  # 预测上涨
            return 1  # 买入信号
        else:  # 预测下跌
            return -1  # 卖出信号
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        if self.model is None:
            print(f"错误：模型 {self.name} 未初始化，无法保存")
            return False
        
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'sequence_length': self.sequence_length,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'feature_scaler': self.feature_scaler,
                'feature_scaler_fitted': getattr(self, 'feature_scaler_fitted', False)
            }
            torch.save(model_state, path)
            
            print(f"模型已保存到 {path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(path):
            print(f"错误：模型文件 {path} 不存在")
            return False
        
        try:
            # 加载模型状态
            model_state = torch.load(path, map_location=self.device)
            
            # 恢复超参数
            self.sequence_length = model_state['sequence_length']
            self.hidden_dim = model_state['hidden_dim']
            self.num_layers = model_state['num_layers']
            self.feature_scaler = model_state['feature_scaler']
            self.feature_scaler_fitted = model_state['feature_scaler_fitted']
            
            # 确定输入特征维度并创建模型
            # 这里我们需要知道输入特征的维度，通过检查特征缩放器的输入形状
            n_features = self.feature_scaler.n_features_in_
            self.model = self.build_model(input_dim=n_features)
            
            # 加载模型权重
            self.model.load_state_dict(model_state['model_state_dict'])
            self.model.eval()  # 设置为评估模式
            
            self.is_trained = True
            print(f"已加载模型 {path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False 