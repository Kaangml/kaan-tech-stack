# Dataclasses & attrs: Daha Az Kodla Daha Çok İş

Bu bölümde, Python'da veri odaklı sınıflar yazma şeklimizi kökten değiştiren `@dataclass` dekoratörünü ve onu inspire eden `attrs` kütüphanesini ele alıyoruz.

## 1. Sorun: "Boilerplate" Kod Cehennemi

Nesne Yönelimli Programlama'da sınıflar sadece karmaşık davranışlar (behavior) için değil, aynı zamanda yapısal verileri bir arada tutmak (data) için de kullanılır. Ancak Python'da, sadece birkaç veri alanını bir arada tutacak basit bir sınıf yazmak bile şaşırtıcı derecede "geveze" olabilir.

Örneğin, 2D bir noktayı temsil eden `Point` sınıfını ele alalım.

```python
# Versiyon 1: Olması gerektiği gibi çalışan, ama yazması zahmetli sınıf
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Anlamlı bir çıktı için __repr__ gerekir
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    # Değer bazlı karşılaştırma için __eq__ gerekir
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
```

Bu kodda yanlış bir şey yok, ama sorunları var:
-   **Çok Fazla Tekrar:** `x` ve `y` attribute'larını `__init__`, `__repr__` ve `__eq__` içinde tekrar tekrar yazdık.
-   **Hata Yapmaya Açık:** Yeni bir `z` attribute'u eklersek, `__init__`, `__repr__` ve `__eq__` metotlarının üçünü de güncellemeyi unutabiliriz. Bu da hatalara yol açar.
-   **Niyet Belirsiz:** Koda bakan biri, bu sınıfın ana amacının ne olduğunu anlamak için tüm bu "boilerplate" (standart, tekrar eden) dunder metotlarını okumak zorundadır.

## 2. Çözüm: `@dataclass` Dekoratörü

Python 3.7 ile standart kütüphaneye eklenen `@dataclass`, bu soruna zarif bir çözüm getirir. O bir **kod üreticisidir**. Siz sadece sınıfınızın veri alanlarını (attribute) ve tiplerini beyan edersiniz, `@dataclass` sizin için arka planda tüm o sıkıcı dunder metotlarını (`__init__`, `__repr__`, `__eq__` vb.) otomatik olarak yazar.

Aynı `Point` sınıfı, `@dataclass` ile:
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```
İşte bu kadar! Bu 3 satırlık kod, yukarıdaki 12 satırlık kodla **tamamen aynı işi yapar.** Daha okunaklı, daha az hataya açık ve niyeti çok daha net.

## 3. Yaygın `@dataclass` Özellikleri

`@dataclass` dekoratörü, davranışını özelleştirmek için bazı parametreler alır:

-   `@dataclass(frozen=True)`: Sınıftan yaratılan nesneleri **değiştirilemez (immutable)** yapar. Bir nesne yaratıldıktan sonra attribute'larını değiştirmeye çalışırsanız `FrozenInstanceError` alırsınız. Bu, veri bütünlüğünü korumak için çok önemlidir.

-   `@dataclass(order=True)`: `__eq__`'e ek olarak, sıralama metotlarını (`__lt__`, `__le__`, `__gt__`, `__ge__`) da otomatik olarak üretir. Bu sayede nesneleri listeler içinde sıralayabilirsiniz.

Ayrıca, `field` fonksiyonu ile her bir attribute'un davranışını daha detaylı kontrol edebiliriz:

-   `password: str = field(repr=False)`: `password` attribute'unun `__repr__` çıktısında (yani `print` edildiğinde) görünmesini engeller. Güvenlik için kritiktir.
-   `items: list = field(default_factory=list)`: Varsayılan değeri mutable (değişebilir) bir tip olan (list, dict gibi) attribute'lar için kullanılır. Her yeni nesne yaratıldığında, `list()` fonksiyonunu çağırarak yeni ve boş bir liste oluşturur, böylece tüm nesnelerin aynı listeyi paylaşması gibi tehlikeli bir durumun önüne geçer.

## 4. Peki ya `attrs`?

`attrs`, `@dataclass`'in ilham kaynağı olan, daha eski ve daha güçlü bir üçüncü parti kütüphanedir. `@dataclass`'in yaptığı her şeyi ve daha fazlasını yapar.
-   **Validator'lar:** Bir attribute'a değer atandığında çalışacak doğrulama kuralları eklemenizi sağlar.
-   **Converter'lar:** Değer atandığında değeri otomatik olarak dönüştürmenizi sağlar (örneğin, string olarak gelen bir tarihi `datetime` nesnesine çevirmek).
-   **Geriye Dönük Uyumluluk:** Eski Python versiyonlarında da çalışır.

Genel kural şudur:
-   Eğer projeniz Python 3.7+ kullanıyorsa ve ekstra güce ihtiyacınız yoksa, standart kütüphanedeki `@dataclass` ile başlayın.
-   Eğer validator, converter gibi gelişmiş özelliklere ihtiyacınız varsa veya daha eski Python versiyonlarını desteklemeniz gerekiyorsa, `attrs` harika bir seçimdir.

Her ikisi de aynı problemi çözer: Sizi "boilerplate" kod yazmaktan kurtararak daha temiz, daha sağlam ve daha modern Python kodu yazmanızı sağlamak. Bu konseptler, `dataclasses_attrs.py` dosyasında kod örnekleriyle pekiştirilmiştir.