# Object Creation, Representation, and Optimization in Python

This document explores the fundamental dunder (double underscore) methods that manage an object's lifecycle, representation, and memory optimization in Python. These methods are the cornerstones of Python's object-oriented structure.

---

### A) __new__ vs. __init__: The Object Lifecycle: Creation

Bir object'in `Class()` çağrısıyla hayata gelmesi iki temel aşamada gerçekleşir: object creation ve initialization.

#### Summary

-   `__new__` **(Creator - The Constructor):** Bir class'ın instance'ı için memory'de yer ayıran ve **boş bir object yaratan** static bir method'dur. Object'in varoluşundan sorumludur.
-   `__init__` **(Initializer - The Initializer):** `__new__` tarafından yaratılmış o boş object'i alır ve başlangıç state'ini ayarlar. Yani, object'i `self.attribute = value` gibi atamalarla "doldurur".

#### When to Use Which?

-   `__init__`: **Neredeyse her zaman.** Bir object'in başlangıç attribute'larını ayarlamak için standart ve en yaygın yoldur.
-   `__new__`: **Object creation'ın *kendisine* müdahale etmek gerektiğinde.** Bu, standart `__init__` akışının yetersiz kaldığı özel ve güçlü senaryolar için bir araçtır.

#### Practical Scenarios (for __new__)

1.  **Singleton:** Uygulama boyunca bir class'tan sadece tek bir instance'ın var olmasını garantilemek için kullanılır.
    -   **AI/ML Use Case:** Büyük bir LLM modelini (örn: bir BERT modeli) veya büyük bir data set'i memory'ye sadece bir kez yüklemek, her seferinde aynı object reference'ını kullanarak memory ve time tasarrufu sağlamak.

2.  **Immutable Classes:** `tuple` gibi immutable base type'lardan miras alındığında, object'in value'ları daha yaratılırken atanmalıdır. Çünkü `__init__` çalıştığında, object'in içi artık değiştirilemez.
    -   **AI/ML Use Case:** Bir deneyin sonuçlarını etkileyen ve asla değişmemesi gereken bir hyperparameter set (`HyperparameterSet`) yaratmak. Bu, deneylerin tekrarlanabilirliği ve reliability'si için kritiktir.

---

### B) __repr__ vs. __str__: The Art of Object Representation

Bu iki method, bir object'in metinsel olarak nasıl temsil edileceğini kontrol eder ve farklı amaçlara hizmet eder.

-   `__repr__` **(For the Developer):** Amacı, **net, teknik ve belirsizliğe yer vermeyen (unambiguous)** bir çıktı üretmektir. İdeal olarak, bu çıktı object'i yeniden yaratabilecek geçerli bir Python code'udur. Debugging, logging ve interactive console için hayatidir.
    -   *Example Output:* `TrainingConfig(lr=0.001, epochs=50, optimizer='adam')`

-   `__str__` **(For the End-User):** Amacı, **okunaklı ve anlaşılır** bir çıktı sunmaktır. `print()` function'ı tarafından kullanılır ve object'in insani bir özetini verir.
    -   *Example Output:* `Eğitim Konfigürasyonu (lr=0.001, epoch=50, optimizer='adam')`

**Hierarchy:** `__str__` tanımlı değilse, `print()` bile `__repr__`'ı kullanır. Bu yüzden her class'ın en azından iyi bir `__repr__`'a sahip olması şiddetle tavsiye edilir.

---

### C) __slots__: Memory Optimization and Control

`__slots__`, varsayılan olarak esneklik için kullanılan `__dict__`'i devre dışı bırakarak object'lerin memory footprint'ini önemli ölçüde azaltma tekniğidir.

#### Why Use It?

Milyonlarca veya milyarlarca küçük, yapısı belirli object yaratıldığında memory tasarrufu sağlamak için kullanılır.

-   **AI/ML Use Case:**
    -   Bir data set'indeki her bir `FeatureVector`.
    -   Büyük bir metin corpus'undaki her bir `Token` object'i.
    -   Bir image'deki her bir `Pixel` object'i.
    -   Parçacık simülasyonlarındaki her bir `Particle`.

#### Trade-off

Esneklik kaybı. `__slots__` ile tanımlanan attribute'lar dışında object'e dinamik olarak `object.new_attribute = value` gibi yeni bir attribute atayamazsınız. Bu bir hata değil, bu optimization'ın getirdiği bilinçli bir kısıtlamadır.