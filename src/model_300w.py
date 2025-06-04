import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np

class WingLoss(tf.keras.losses.Loss):
    """Wing Loss for facial landmark detection"""
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        
    def call(self, y_true, y_pred):
        # Reshape if needed
        y_true = tf.reshape(y_true, [-1, 68, 2])
        y_pred = tf.reshape(y_pred, [-1, 68, 2])
        
        # Calculate L2 distance
        diff = y_true - y_pred
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-6)
        
        # Wing loss calculation
        losses = tf.where(
            dist < self.omega,
            self.omega * tf.math.log(1 + dist / self.epsilon),
            dist - self.C
        )
        
        return tf.reduce_mean(losses)

class AdaptiveWingLoss(tf.keras.losses.Loss):
    """Adaptive Wing Loss - even better for landmarks"""
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        # Reshape
        y_true = tf.reshape(y_true, [-1, 68, 2])
        y_pred = tf.reshape(y_pred, [-1, 68, 2])
        
        # Calculate pixel-wise difference
        delta = tf.abs(y_pred - y_true)
        
        # Adaptive factor
        A = self.omega * (1 / (1 + tf.pow(self.theta / self.epsilon, self.alpha - y_true))) * \
            (self.alpha - y_true) * (tf.pow(self.theta / self.epsilon, self.alpha - y_true - 1)) * \
            (1 / self.epsilon)
        
        C = self.theta * A - self.omega * tf.log(1 + tf.pow(self.theta / self.epsilon, self.alpha - y_true))
        
        losses = tf.where(
            delta < self.theta,
            self.omega * tf.log(1 + tf.pow(delta / self.epsilon, self.alpha - y_true)),
            A * delta - C
        )
        
        return tf.reduce_mean(losses)

class FaceLandmark300W(Model):
    """Improved model using MobileNetV2 backbone + custom head"""
    
    def __init__(self, input_shape=(224, 224, 3), use_pretrained=True):
        super(FaceLandmark300W, self).__init__()
        
        # Use MobileNetV2 as backbone (lightweight and effective)
        self.backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet' if use_pretrained else None,
            pooling='avg'
        )
        
        # Fine-tuning: unfreeze last 30 layers
        for layer in self.backbone.layers[:-30]:
            layer.trainable = False
            
        # Custom head for landmark regression
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Multi-scale feature extraction
        self.conv_1x1 = layers.Conv2D(256, (1, 1), activation='relu')
        
        # Regression head with residual connections
        self.dense1 = layers.Dense(1024, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(512, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        
        self.dense3 = layers.Dense(256, activation='relu')
        self.bn3 = layers.BatchNormalization()
        
        # Output layer - 68 landmarks * 2 coordinates = 136
        # Using tanh to constrain output to [-1, 1] range
        self.output_layer = layers.Dense(136, activation='tanh')
        
    def call(self, inputs, training=False):
        # Extract features from backbone
        features = self.backbone(inputs, training=training)
        
        # Head processing
        x = self.dense1(features)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        
        # Output landmarks (normalized to [-1, 1])
        landmarks_flat = self.output_layer(x)
        
        # Reshape to (batch_size, 68, 2)
        landmarks = tf.reshape(landmarks_flat, (-1, 68, 2))
        
        return landmarks

class HourglassBlock(layers.Layer):
    """Hourglass block for better spatial understanding"""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        
    def build(self, input_shape):
        # Down path
        self.down1 = layers.Conv2D(self.num_features, 3, strides=2, padding='same', activation='relu')
        self.down2 = layers.Conv2D(self.num_features, 3, strides=2, padding='same', activation='relu')
        
        # Middle
        self.middle = layers.Conv2D(self.num_features, 3, padding='same', activation='relu')
        
        # Up path
        self.up1 = layers.Conv2DTranspose(self.num_features, 3, strides=2, padding='same', activation='relu')
        self.up2 = layers.Conv2DTranspose(self.num_features, 3, strides=2, padding='same', activation='relu')
        
    def call(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        # Middle
        m = self.middle(d2)
        
        # Decoder with skip connections
        u1 = self.up1(m) + d1
        u2 = self.up2(u1) + x
        
        return u2

class FaceLandmarkHourglass(Model):
    """Advanced model using stacked hourglass architecture"""
    
    def __init__(self, num_stacks=2, num_features=128):
        super().__init__()
        self.num_stacks = num_stacks
        
        # Initial processing
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(num_features, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        
        # Stacked hourglass modules
        self.hourglasses = [HourglassBlock(num_features) for _ in range(num_stacks)]
        
        # Output heads for each stack
        self.outputs = []
        for _ in range(num_stacks):
            self.outputs.append(layers.Conv2D(68, 1, padding='same'))  # 68 heatmaps
            
        # Final regression
        self.final_conv = layers.Conv2D(256, 1, activation='relu')
        self.global_pool = layers.GlobalAveragePooling2D()
        self.final_dense = layers.Dense(136, activation='tanh')
        
    def call(self, inputs, training=False):
        # Initial processing
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Process through hourglasses
        heatmaps = []
        for i in range(self.num_stacks):
            x = self.hourglasses[i](x)
            heatmap = self.outputs[i](x)
            heatmaps.append(heatmap)
            
            if i < self.num_stacks - 1:
                x = x + heatmap  # Intermediate supervision
                
        # Convert final heatmap to coordinates
        final_features = self.final_conv(heatmaps[-1])
        pooled = self.global_pool(final_features)
        landmarks_flat = self.final_dense(pooled)
        
        landmarks = tf.reshape(landmarks_flat, (-1, 68, 2))
        
        return landmarks

# Helper function to create model with custom loss
def create_model(model_type='mobilenet', input_shape=(224, 224, 3), loss_type='wing'):
    """Create model with specified architecture and loss"""
    
    # Choose model
    if model_type == 'mobilenet':
        model = FaceLandmark300W(input_shape=input_shape)
    elif model_type == 'hourglass':
        model = FaceLandmarkHourglass()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Choose loss
    if loss_type == 'wing':
        loss = WingLoss()
    elif loss_type == 'adaptive_wing':
        loss = AdaptiveWingLoss()
    elif loss_type == 'mse':
        loss = 'mse'
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    
    return model