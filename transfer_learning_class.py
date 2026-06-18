"""
Transfer Learning Model Uygulaması. Hazır modelin ağırlıkarını kullanarak bir sınıflandırıcı uygulaması.
Hiperparametreler config.py içerisindedir.
@ NURULLAH ALİ AKA tarafından hazırlanmıştır.
"""
import torch
from torch import nn
from torchvision import models

class TransferLearningModel(nn.Module):

    def __init__(self, num_classes: int, backbone_name: str = "resnet50"):
        super().__init__()

        # 1. Hazır modeli ağırlıklarıyla yükle (Güncel API kullanımı)
        weights = models.get_model_weights(backbone_name).DEFAULT
        self.backbone = models.get_model(backbone_name, weights=weights)

        # 2. Omurgayı (Backbone) Dondur
        self._freeze_backbone()

        # 3. Giriş özellik sayısını dinamik ve güvenli şekilde al
        # ResNet için 'fc', EfficientNet/MobileNet/VGG için 'classifier' kullanılır.
        """hasattr: "Bu nesnenin içinde, belirttiğim isimde bir özellik (attribute), değişken veya fonksiyon var mı, yok mu?" sorusunu sorar."""
        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            # Orijinal katmanı boşa çıkarıp, kendi dinamik başlığımızı ekliyoruz
            self.backbone.fc = nn.Identity()
            """nn.Identity() bir "Boş Geçiş" katmanıdır. 
            Kendisine gelen tensöre hiçbir matematiksel işlem yapmadan, 
            ağırlık (weight) veya bias eklemeden aynen bir sonraki katmana iletir."""

        elif hasattr(self.backbone, "classifier"):
            # Eğer bir zincirse (Sequential), içindeki Linear katmanını ara ve bul
            if isinstance(self.backbone.classifier, nn.Sequential):
                linear_layer = None
                for layer in self.backbone.classifier:
                    if isinstance(layer, nn.Linear):
                        linear_layer = layer
                        break

                if linear_layer is None:
                    raise AttributeError(
                        "Classifier zinciri içinde nn.Linear katmanı bulunamadı!"
                    )

                in_features = linear_layer.in_features
            else:
                # Eğer Sequential değil de direkt tek bir Linear katmanıysa
                in_features = self.backbone.classifier.in_features

            # Orijinal classifier'ı bypass et
            self.backbone.classifier = nn.Identity()

        else:
            raise NotImplementedError(
                f"{backbone_name} mimarisi için özel başlık konumu tanımlanmalıdır."
            )

        # 4. Profesyonel Dinamik Başlık (Dynamic Head) Tasarımı
        # Direkt tek katman yerine Dropout ve BatchNormalization içeren esnek bir yapı
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def _freeze_backbone(self):
        """Omurgadaki tüm parametrelerin gradyan güncellemelerini kapatır."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Özellik çıkarımı (Feature Extraction) sabit omurgadan geçer

        # Sadece sınıflandırıcı başlık eğitilir
        logits = self.classifier_head(self.backbone(x))
        return logits