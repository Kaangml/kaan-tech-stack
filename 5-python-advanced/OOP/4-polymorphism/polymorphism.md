A. class_and_object/

Konular (advanced)
	•	Python object model: __new__, __init__, __repr__, __str__, __slots__.
	•	Descriptor protokolü (property, staticmethod, classmethod, custom descriptor).
	•	Metaclasses (use-cases: otomatik kayıt, registry pattern, otomatik validation).
	•	Dataclasses & attrs (frozen, validation, factory, asdict).
	•	Immutable objects & copy semantics (deep vs shallow).
	•	Context managers & resource handling (__enter__, __exit__, contextlib).

ML/AI pratik karşılıkları
	•	__slots__ ile hafıza optimizasyonu büyük modellerde veya milyonlarca küçük objede faydalı.
	•	Descriptor/metaclass ile otomatik model/optimizer/register yapıları (ör: plugin registry).
	•	Dataclasses kullanarak configuration/experiment metadata saklama (serializable, reproducible).

Dosya önerisi
	•	class_and_object.md (konu + kısa örnekler + nerede kullanılır)
	•	examples/
	•	model_registry.py (metaclass veya decorator-based registry)
	•	config_dataclass.py (dataclass-based config with validation)
	•	class_and_object.ipynb (live demo: registry + küçük toy model)
	•	tests/test_registry.py

Örnek uygulama fikri
	•	Model Registry: tüm nn.Module türevlerini otomatik kaydeden bir metaclass ve bu registry üzerinden model seçim factory’si.