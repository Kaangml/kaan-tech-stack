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

1-class_and_object/
	1-objects
		objects.md
	2-descriptor
		descriptor.md
	3-metaclass
		metaclass.md
	4-dataclass_attrs
		dataclass_attrs.md