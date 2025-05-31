# 第五章 集成学习优化

本章详细介绍系统的集成学习优化策略。通过设计多个基础模型变体并采用优化的集成方法，进一步提升了水军检测的性能。

## 5.1 基础模型设计

### 5.1.1 模型变体设计

系统实现了四种基础模型变体，每种变体都针对特定的特征组合进行了优化：

1. **文本增强型模型**
```python
class TextEnhancedModel(BaseModel):
    def __init__(self, config):
        super(TextEnhancedModel, self).__init__(config)
        # 增强文本特征提取
        self.text_encoder = EnhancedTextEncoder(
            bert_model=config.BERT_MODEL,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS
        )
        
        # 注意力机制
        self.text_attention = MultiHeadAttention(
            config.HIDDEN_SIZE,
            config.NUM_HEADS
        )
        
    def forward(self, inputs):
        # 文本特征提取
        text_features = self.text_encoder(
            inputs['text_ids'],
            inputs['attention_mask']
        )
        
        # 注意力处理
        text_attended = self.text_attention(
            text_features,
            text_features,
            text_features
        )
        
        # 其他特征处理
        other_features = self.process_other_features(inputs)
        
        # 特征融合
        combined = self.fusion_layer(
            torch.cat([text_attended, other_features], dim=-1)
        )
        
        return self.classifier(combined)
```

2. **行为增强型模型**
```python
class BehaviorEnhancedModel(BaseModel):
    def __init__(self, config):
        super(BehaviorEnhancedModel, self).__init__(config)
        # 增强行为特征提取
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.TEMPORAL_DIM,
            hidden_dim=config.HIDDEN_DIM
        )
        
        self.interaction_encoder = InteractionEncoder(
            input_dim=config.INTERACTION_DIM,
            hidden_dim=config.HIDDEN_DIM
        )
        
    def forward(self, inputs):
        # 时序特征处理
        temporal_features = self.temporal_encoder(
            inputs['temporal_data']
        )
        
        # 互动特征处理
        interaction_features = self.interaction_encoder(
            inputs['interaction_data']
        )
        
        # 特征融合
        behavior_features = torch.cat([
            temporal_features,
            interaction_features
        ], dim=-1)
        
        # 其他特征处理
        other_features = self.process_other_features(inputs)
        
        # 最终融合
        combined = self.fusion_layer(
            torch.cat([behavior_features, other_features], dim=-1)
        )
        
        return self.classifier(combined)
```

3. **社交网络增强型模型**
```python
class SocialEnhancedModel(BaseModel):
    def __init__(self, config):
        super(SocialEnhancedModel, self).__init__(config)
        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            GATConv(config.NODE_FEATURES, config.HIDDEN_DIM),
            GATConv(config.HIDDEN_DIM, config.HIDDEN_DIM)
        ])
        
        # 图池化层
        self.graph_pooling = GlobalAttentionPooling(
            config.HIDDEN_DIM
        )
        
    def forward(self, inputs):
        # 图特征提取
        x, edge_index = inputs['node_features'], inputs['edge_index']
        
        # GNN处理
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        
        # 图池化
        graph_features = self.graph_pooling(x, inputs['batch'])
        
        # 其他特征处理
        other_features = self.process_other_features(inputs)
        
        # 特征融合
        combined = self.fusion_layer(
            torch.cat([graph_features, other_features], dim=-1)
        )
        
        return self.classifier(combined)
```

4. **平衡型模型**
```python
class BalancedModel(BaseModel):
    def __init__(self, config):
        super(BalancedModel, self).__init__(config)
        # 特征提取器
        self.feature_extractors = nn.ModuleDict({
            'text': TextEncoder(config),
            'behavior': BehaviorEncoder(config),
            'social': SocialEncoder(config)
        })
        
        # 注意力融合
        self.attention_fusion = MultiViewAttentionFusion(
            config.FEATURE_DIM,
            config.NUM_VIEWS
        )
        
    def forward(self, inputs):
        # 提取各类特征
        features = {
            name: extractor(inputs[f'{name}_data'])
            for name, extractor in self.feature_extractors.items()
        }
        
        # 注意力融合
        fused = self.attention_fusion(list(features.values()))
        
        return self.classifier(fused)
```

### 5.1.2 模型训练策略

为提高基础模型的性能，采用了以下训练策略：

1. **损失函数设计**
```python
def calculate_loss(self, outputs, targets):
    # 分类损失
    ce_loss = F.cross_entropy(outputs, targets)
    
    # 正则化损失
    l2_loss = 0.0
    for param in self.parameters():
        l2_loss += torch.norm(param, p=2)
    
    # 特征一致性损失
    consistency_loss = self.feature_consistency_loss(outputs)
    
    # 总损失
    total_loss = ce_loss + self.config.L2_WEIGHT * l2_loss + \
                 self.config.CONSISTENCY_WEIGHT * consistency_loss
    
    return total_loss
```

2. **学习率调度**
```python
def configure_optimizers(self):
    # 优化器
    optimizer = AdamW(
        self.parameters(),
        lr=self.config.LEARNING_RATE,
        weight_decay=self.config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.config.WARMUP_STEPS,
        num_training_steps=self.config.TOTAL_STEPS
    )
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step'
        }
    }
```

## 5.2 集成策略

### 5.2.1 模型融合方法

系统实现了多种模型融合策略：

1. **投票集成**
```python
class VotingEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
    def predict(self, inputs):
        predictions = []
        
        # 获取各模型预测
        for model, weight in zip(self.models, self.weights):
            pred = model(inputs)
            pred = F.softmax(pred, dim=-1)
            predictions.append(pred * weight)
        
        # 加权投票
        ensemble_pred = sum(predictions)
        
        return ensemble_pred
```

2. **Stacking集成**
```python
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, meta_model):
        super(StackingEnsemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        
    def forward(self, inputs):
        # 基础模型预测
        base_preds = []
        for model in self.base_models:
            pred = model(inputs)
            base_preds.append(pred)
        
        # 合并预测结果
        stacked = torch.stack(base_preds, dim=1)
        
        # 元模型预测
        final_pred = self.meta_model(stacked)
        
        return final_pred
```

3. **Bagging集成**
```python
class BaggingEnsemble:
    def __init__(self, base_model, num_models):
        self.models = []
        for _ in range(num_models):
            # 创建模型副本
            model = copy.deepcopy(base_model)
            self.models.append(model)
            
    def train(self, train_data):
        for model in self.models:
            # 随机采样训练数据
            sampled_data = self.bootstrap_sample(train_data)
            # 训练单个模型
            model.train(sampled_data)
            
    def predict(self, inputs):
        predictions = []
        
        # 收集所有模型的预测
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        # 平均预测结果
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_pred
```

### 5.2.2 集成优化策略

为进一步提升集成模型的性能，实现了以下优化策略：

1. **动态权重调整**
```python
def adjust_ensemble_weights(self, val_metrics):
    # 计算每个模型的性能分数
    scores = []
    for metric in val_metrics:
        score = (
            metric['accuracy'] * self.config.ACC_WEIGHT +
            metric['f1'] * self.config.F1_WEIGHT +
            metric['auc'] * self.config.AUC_WEIGHT
        )
        scores.append(score)
    
    # 更新权重
    total_score = sum(scores)
    new_weights = [score/total_score for score in scores]
    
    return new_weights
```

2. **预测校准**
```python
def calibrate_predictions(self, logits, temperature=1.0):
    # 温度缩放
    scaled_logits = logits / temperature
    
    # 计算校准后的概率
    probabilities = F.softmax(scaled_logits, dim=-1)
    
    return probabilities
```

3. **模型选择策略**
```python
def select_ensemble_models(self, candidates, num_select):
    # 计算模型间的相关性
    correlations = self.calculate_prediction_correlation(candidates)
    
    # 贪心选择互补性强的模型
    selected = []
    while len(selected) < num_select:
        best_candidate = None
        min_correlation = float('inf')
        
        for candidate in candidates:
            if candidate in selected:
                continue
            
            # 计算与已选模型的平均相关性
            mean_corr = np.mean([
                correlations[candidate][selected_model]
                for selected_model in selected
            ]) if selected else 0
            
            if mean_corr < min_correlation:
                min_correlation = mean_corr
                best_candidate = candidate
        
        selected.append(best_candidate)
    
    return selected
```

## 5.3 本章小结

本章详细介绍了系统的集成学习优化策略。通过设计多个针对性的基础模型变体，并采用优化的集成方法，显著提升了水军检测的性能。主要创新点包括：

1. 设计了四种互补的基础模型：
   - 文本增强型模型
   - 行为增强型模型
   - 社交网络增强型模型
   - 平衡型模型

2. 实现了多种集成策略：
   - 投票集成
   - Stacking集成
   - Bagging集成

3. 优化了集成学习效果：
   - 动态权重调整
   - 预测校准
   - 模型选择策略

通过这些优化策略的综合应用，系统在准确性和鲁棒性方面都取得了显著提升，为实际应用提供了可靠的技术支持。 