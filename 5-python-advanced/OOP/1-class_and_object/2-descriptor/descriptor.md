# Python Descriptors: Nesne Modelinin Çekirdek Mekaniği

Bu bölümde, Python'un en güçlü ama en az anlaşılan özelliklerinden biri olan descriptor'ları inceleyeceğiz. Descriptor'lar, "attribute access interception" (özellik erişimini yakalama) için bir mekanizma sunar ve Python'daki `property`, metotlar, `staticmethod` ve `classmethod` gibi birçok temel yapının arkasındaki teknolojidir.

## 1. "Descriptor = Attribute Access Interception" Ne Demek?

Bir Python objesinde `obj.x` gibi bir attribute erişimi yaptığınızda, Python doğrudan "objenin x belleğindeki değeri oku" gibi basit bir işlem yapmaz. Bunun yerine, tüm attribute erişimleri merkezi bir mekanizmadan geçer. Bu mekanizma, `type(obj).__getattribute__(obj, "x")` metodudur.

**Interception (Yakalama):** İşte descriptor'lar tam bu noktada devreye girer. Eğer erişilmek istenen attribute (`x`), sıradan bir değer yerine `__get__`, `__set__` veya `__delete__` metotlarından en az birini içeren bir "descriptor" objesi ise, Python bu attribute'a erişimi bu özel metotlar üzerinden yönlendirir:

-   **Okuma:** `obj.x` -> `descriptor.__get__(obj, type(obj))`
-   **Yazma:** `obj.x = value` -> `descriptor.__set__(obj, value)`
-   **Silme:** `del obj.x` -> `descriptor.__delete__(obj)`

Bu sayede, bir attribute'a yapılan her türlü erişim, doğrudan veriye ulaşmadan önce "yakalanabilir" ve özel bir mantıkla yönetilebilir.

## 2. Descriptor Protokolünün Kuralları

### Kural 1: Descriptor Class Attribute Olmak Zorundadır

Python, descriptor mekanizmasını yalnızca bir class'ın `__dict__`'i içinde arar. Bir objenin instance'ına (örneğine) sonradan eklenen bir descriptor objesi aktive olmaz.

```python
class MyDescriptor:
    # ... descriptor metotları ...

class MyClass:
    # Class attribute olarak tanımlandığı için BU BİR DESCRIPTOR'DIR
    x = MyDescriptor()

    def __init__(self):
        # Instance attribute olduğu için BU BİR DESCRIPTOR DEĞİLDİR
        self.y = MyDescriptor()
```

**Neden?** Çünkü descriptor'lar, veriyi değil, o veriye erişim politikasını temsil eder. Bu politika, o class'ın tüm instance'ları için ortak ve tutarlı olmalıdır. Eğer her instance kendi erişim kurallarını tanımlasaydı, Python'un nesne modeli kaotik ve öngörülemez olurdu.

### Kural 2: Data ve Non-Data Descriptor Ayrımı

Bu, attribute erişim sırasını anlamak için en kritik ayrımdır ve ayrım tamamen `__set__` veya `__delete__` metodunun varlığına bağlıdır.

#### Non-Data Descriptor
Yalnızca `__get__` metoduna sahiptir.
-   **Amacı:** Genellikle "hesaplanmış" değerler üretmek veya bir instance'a "bağlanmış" (bound) bir davranış sunmak içindir.
-   **En İyi Örnek: Metotlar.** Bir class içine yazdığınız her `def` bir non-data descriptor'dır. `my_obj.my_method` dediğinizde, fonksiyonun `__get__` metodu çağrılır, bu metot fonksiyonu `my_obj` instance'ına bağlayarak size "bound method" adı verilen çağrılabilir bir nesne döndürür. Bir metoda dışarıdan değer atayamayacağınız için (`my_obj.my_method = 5` metodu değiştirmez, sadece instance `__dict__`'ine yazar), `__set__` metoduna ihtiyaçları yoktur.

#### Data Descriptor
`__get__` metoduna ek olarak `__set__` ve/veya `__delete__` metotlarından en az birine sahiptir.
-   **Amacı:** Adından da anlaşılacağı gibi, bir "veri" parçasının yönetimini üstlenir. Bir attribute'a değer atandığında (set edildiğinde) bir doğrulama (validation) yapmak, veritabanına kaydetmek veya bir log tutmak gibi yan etkiler oluşturmak için kullanılır. Değer atama sürecini kontrol altına aldığı için `__set__` metodu kritik öneme sahiptir.
-   **Örnekler:** `@property` en bilinen örneğidir. ORM (Object-Relational Mapper) kütüphanelerindeki `Field`'lar (örneğin Django'daki `models.CharField`) ve bizim yazdığımız `Typed` descriptor'ı da mükemmel data descriptor örnekleridir.

## 3. Attribute Erişim Öncelik Sırası (Resolution Order)

Python `obj.x` gördüğünde, değeri bulmak için şu deterministik sırayı izler:

1.  **Data Descriptor:** Class'ın (`type(obj)`) `__dict__`'inde `x` isminde bir data descriptor var mı? Varsa, onun `__get__` metodu çağrılır ve işlem biter. **Bu noktada instance'ın `__dict__`'i tamamen yok sayılır.** Bu en yüksek önceliktir.
2.  **Instance Attribute:** `obj`'nin `__dict__`'inde `x` anahtarı var mı? Varsa, oradaki değer doğrudan döndürülür.
3.  **Non-Data Descriptor:** Class'ın `__dict__`'inde `x` isminde bir non-data descriptor var mı? Varsa, onun `__get__` metodu çağrılır.
4.  **Class Attribute:** Class'ın `__dict__`'inde `x` isminde (descriptor olmayan) bir attribute var mı? Varsa, o değer döndürülür.
5.  **Fallback:** Yukarıdakilerin hiçbiri başarılı olmazsa, son çare olarak `obj` üzerinde `__getattr__` metodu aranır ve varsa çağrılır.

#### "Aha!" Anı: Bu Sıra Neden Önemli?
Bu sıra, "bir instance attribute'u neden bir metodu gölgelerken, bir `property`'yi gölgeleyemez?" sorusunun cevabıdır. Çünkü metotlar non-data descriptor'dır (3. sıra), `property` ise data descriptor'dır (1. sıra).

Bu sıralama, şu meşhur sorunun cevabıdır: **"Neden bir instance attribute'u bir metodu 'ezebilirken' (shadowing), bir `@property`'yi ezemez?"**

-   **Metodu Ezmek (Başarılı):** Bir metot, bir **non-data descriptor**'dır (Öncelik #3). Siz `obj.my_method = 10` dediğinizde, `obj`'nin `__dict__`'ine (`'my_method': 10`) kaydını eklersiniz. Bu işlem, **Öncelik #2**'ye karşılık gelir. Bir sonraki `obj.my_method` erişiminizde, Python 2. adımı 3. adımdan *önce* kontrol ettiği için, `__dict__`'teki `10` değerini bulur ve non-data descriptor'a hiç ulaşmadan onu döndürür. Metot "gölgelenmiş" olur.

-   **Property'yi Ezmek (Başarısız):** Bir `@property`, bir **data descriptor**'dır (Öncelik #1). Siz `obj.age = 25` gibi bir atama yaptığınızda, Python daha 1. adımda class üzerinde bir data descriptor olduğunu görür ve hemen onun `__set__` metodunu çağırır. `obj.__dict__`'e doğrudan bir yazma işlemi yapılmasına asla izin verilmez. Okuma yaparken de yine 1. adım (`__get__`) devreye girer ve `obj.__dict__`'teki olası bir değer by-pass edilir. Bu yüzden data descriptor'lar instance attribute'ları tarafından ezilemez.

## 4. `@property`: Yaygın Bir Data Descriptor

`@property` bir "syntactic sugar" (daha kısa yazım) değildir. Arka planda bambaşka bir nesne modeli işleten, tam teşekküllü bir data descriptor'dır.

Şu kod:
```python
class MyClass:
    @property
    def x(self):
        return self._x
```
Aslında şunun kısa halidir:
```python
class MyClass:
    def x(self):
        return self._x
    x = property(fget=x) # 'property' class'ından bir instance oluşturuluyor
```
`property` class'ı `__get__`, `__set__` ve `__delete__` metotlarını barındırdığı için bir data descriptor'dır ve bu sayede attribute erişimini yönetir. Özellikle "validation gateway" (doğrulama geçidi) olarak kullanılarak encapsulation (kapsülleme) sağlar.

## 5. `__set_name__`: Modern Descriptor'ların Anahtarı

Python 3.6 ile gelen `__set_name__` metodu, bir descriptor'ın, sahibi olan class oluşturulurken kendi attribute ismini öğrenmesini sağlar.

`__get__` metodunun `instance` parametresi bu noktada kritik bir rol oynar. Eğer attribute bir instance üzerinden (`obj.x`) çağrılırsa, `__get__` metoduna gelen `instance` parametresi `obj`'nin kendisidir. Eğer attribute class üzerinden (`MyClass.x`) çağrılırsa, bu `instance` parametresi `None` olur. Bu ayrım, descriptor'ın, çağrıldığı bağlama göre (instance erişimi mi, yoksa class erişimi mi) farklı davranmasına olanak tanır. Örneğin, class üzerinden erişildiğinde descriptor'ın kendisini döndürmesi yaygın bir pratiktir.

```python
class MyDescriptor:
    def __set_name__(self, owner_class, name):
        self.public_name = name
        self.private_name = '_' + name
```
Bu metot sayesinde, descriptor'lar artık yeniden kullanılabilir hale gelir. Her seferinde ismini hard-code etmek yerine, atandığı attribute'un ismini dinamik olarak öğrenir. Bu, ORM (Object-Relational Mapper) kütüphanelerindeki `Field`'ların veya Pydantic modellerinin temelini oluşturur.

## 6. Pratik Uygulama

Teoriyi pratiğe dökmek için `descriptor.py` dosyasında somut örnekler hazırladık. Bu dosyada şunları bulabilirsiniz:

-   Data ve Non-Data descriptor'ların öncelik sırasını kanıtlayan bir test.
-   `__set_name__` kullanarak tip kontrolü yapan `Typed` isminde yeniden kullanılabilir bir descriptor.
-   Bu `Typed` descriptor'ını kullanarak basit bir ORM `Field`'ını simüle eden bir yapı.

Bu örnekler, descriptor'ların Python nesne modelini nasıl şekillendirdiğini ve gelişmiş framework'lerin temelini nasıl oluşturduğunu göstermektedir.