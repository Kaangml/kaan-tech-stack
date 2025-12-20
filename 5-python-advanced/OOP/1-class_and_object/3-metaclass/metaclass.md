# Metaclass: Sınıfları Yaratan Sınıflar

Bu bölümde, Python'un en derin ve en güçlü konseptlerinden biri olan "metaclass"leri inceleyeceğiz. Eğer descriptor'lar nesnelerin *attribute erişimini* kontrol ediyorsa, metaclass'ler de **sınıfların yaratılışını** kontrol eder.

## 1. Python'da Her Şey Bir Nesnedir (Sınıflar Dahil)

Python'da bir `int`, `str` veya kendi yazdığımız `MyObject` ne ise, `class MyClass:` ifadesiyle oluşturduğumuz `MyClass`'ın kendisi de odur: **bir nesne**.

Her nesnenin bir tipi vardır. `type(5)` -> `<class 'int'>` der. Peki bir sınıfın tipi nedir?

```python
class MyClass:
    pass

print(type(MyClass))  # Çıktı: <class 'type'>
```

İşte bu, ilk ve en önemli adımdır. `MyClass` adını verdiğimiz nesnenin kendisi, `type` sınıfının bir instance'ıdır. Bu durumda `type`, `MyClass`'ın **metaclass**'idir. `type`, Python'daki varsayılan (default) metaclass'tir.

## 2. Sınıf Yaratımına Müdahale Etmek

"Bir sınıfın yaratılmasına nasıl müdahale ederiz?" sorusunun cevabı, `type`'ın nasıl çalıştığını anlamaktan geçer. `class` anahtar kelimesi aslında bir "syntactic sugar"dır. Python yorumlayıcısı şu kodu gördüğünde:

```python
class MySpam(Eggs):
    ham = 42
```

Arka planda aslında `type`'ı bir sınıf yaratıcı olarak çağırır:

```python
MySpam = type('MySpam', (Eggs,), {'ham': 42})
```

`type(name, bases, attrs)` çağrısı üç argüman alır:
1.  `name`: Sınıfın adı (`str`).
2.  `bases`: Sınıfın miras alacağı base class'ların bir tuple'ı.
3.  `attrs`: Sınıfın attribute ve metotlarını içeren bir dictionary.

İşte bu noktada metaclass'ler devreye girer. Eğer biz `type` yerine kendi özel "sınıf yaratıcımızı" devreye sokabilirsek, sınıf yaratılma anının her adımını (isim, miras yapısı, attribute'lar) kontrol edebiliriz.

## 3. Custom Metaclass Nasıl Yazılır?

Custom bir metaclass yazmak için `type`'tan miras alan bir sınıf oluştururuz ve onun `__new__` veya `__init__` metotlarını override ederiz.

-   `__new__(mcs, name, bases, attrs)`: Bu metot, sınıf nesnesi yaratılmadan **önce** çağrılır. `mcs` burada metaclass'in kendisidir. Bu metot içinde `name` (sınıf adı), `bases` (miras listesi) ve `attrs` (attribute'lar) üzerinde değişiklik yapabiliriz. Sınıfı gerçekten yaratmak için en sonunda `super().__new__(mcs, name, bases, attrs)` çağrısını yapıp yeni sınıf nesnesini döndürmeliyiz. **Asıl manipülasyon burada yapılır.**

-   `__init__(cls, name, bases, attrs)`: Bu metot, sınıf nesnesi `__new__` tarafından yaratıldıktan **sonra** çağrılır. `cls` burada yeni yaratılmış olan sınıfın kendisidir. Bu metot, yaratılmış olan sınıf üzerinde ek başlatma veya konfigürasyon işlemleri yapmak için kullanılır.

Bir sınıfın bu metaclass'i kullanmasını sağlamak için Python 3'te şu sözdizimi kullanılır:

```python
class MyClass(metaclass=MyMetaclass):
    ...
```

## 4. Pratik Kullanım: Otomatik Kayıt (Registry) Sistemi

Framework'lerin en yaygın kullandığı metaclass senaryosu, belirli bir tipteki sınıfları (modeller, plugin'ler, serializer'lar vb.) otomatik olarak bir yere kaydetmektir.

Örneğin, bir plugin sistemimiz olduğunu ve `BasePlugin`'den türeyen tüm alt sınıfları otomatik olarak bir sözlüğe kaydetmek istediğimizi varsayalım.

1.  `PluginMeta` adında bir metaclass oluşturacağız.
2.  Bu metaclass, içinde `_plugins` adında bir dictionary tutacak.
3.  `PluginMeta`'nın `__new__` veya `__init__` metodu, yeni bir sınıf yaratıldığında, o sınıfı isminden `_plugins` sözlüğüne ekleyecek.
4.  `BasePlugin` sınıfı, `metaclass=PluginMeta` kullanarak bu mekanizmayı aktif hale getirecek.
5.  Artık herhangi bir geliştirici `BasePlugin`'den yeni bir sınıf türettiğinde, o sınıf otomatik olarak bizim plugin listemize eklenecek, hiçbir ek `register()` çağrısı yapmasına gerek kalmadan!

Bu güçlü desen, bir framework'ün, kullanıcı tarafından yazılan kod hakkında "bilgi sahibi" olmasını sağlar. Django'nun, siz tanımlar tanımlamaz modellerinizi bilmesi veya bir REST framework'ünün tüm API endpoint'lerinizi tanıması bu sayede olur.

Teoriyi pratiğe dökmek için `metaclass.py` dosyasında bu plugin kayıt sistemini adım adım kodlayacağız.