# -*- coding: utf-8 -*- 

"""
Bu dosya, descriptor.md dosyasında anlatılan Python descriptor mekanizmalarını
somut kod örnekleriyle göstermek için hazırlanmıştır.
"""

print("--- 1. Data ve Non-Data Descriptor Öncelik Farkı ---")

# Data Descriptor: __set__ metoduna sahip
class DataDescriptor:
    def __get__(self, instance, owner):
        print(f"DataDescriptor: __get__ çağrıldı (instance değeri yok sayılacak)")
        return "data descriptor değeri"

    def __set__(self, instance, value):
        print(f"DataDescriptor: __set__ çağrıldı, değer: {value}")
        # Normalde burada bir yere kaydedilir, ama örnek için geçiyoruz.
        pass

# Non-Data Descriptor: sadece __get__ metoduna sahip
class NonDataDescriptor:
    def __get__(self, instance, owner):
        print(f"NonDataDescriptor: __get__ çağrıldı")
        return "non-data descriptor değeri"

class ResolutionOrderTest:
    data_desc = DataDescriptor()
    non_data_desc = NonDataDescriptor()

# Test objesi oluşturalım
test_obj = ResolutionOrderTest()

# Instance'ların __dict__'ine aynı isimde değerler atayalım
test_obj.__dict__['data_desc'] = "instance değeri (data)"
test_obj.__dict__['non_data_desc'] = "instance değeri (non-data)"

print("\nData descriptor erişimi testi:")
# Beklenen: Data descriptor'ın __get__'i çağrılır, instance değeri ezilir.
print(f"Sonuç: {test_obj.data_desc}")

print("\nNon-data descriptor erişimi testi:")
# Beklenen: Instance __dict__'indeki değer, non-data descriptor'ı ezer.
print(f"Sonuç: {test_obj.non_data_desc}")


print("\n\n--- 2. __set_name__ ile Yeniden Kullanılabilir Descriptor ---")

class Typed:
    """
    Belirli bir tipi zorunlu kılan bir data descriptor.
    __set_name__ sayesinde hangi attribute'a atandığını bilir ve
    değeri instance'ın __dict__'inde saklar.
    """
    def __init__(self, expected_type):
        self.type = expected_type
        self.public_name = None
        self.private_name = None

    def __set_name__(self, owner, name):
        # Bu descriptor bir class'a atandığında otomatik olarak çağrılır.
        print(f"Typed descriptor '{name}' attribute'una atandı.")
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, instance, owner):
        if instance is None:
            # Class üzerinden erişim (örn: Person.age)
            return self

        # private name kullanarak instance'ın __dict__'inden değeri al
        value = instance.__dict__.get(self.private_name)
        print(f"'{self.public_name}' okunuyor, değer: {value}")
        return value

    def __set__(self, instance, value):
        if not isinstance(value, self.type):
            raise TypeError(f"'{self.public_name}' için {self.type.__name__} tipinde değer bekleniyordu, "
                            f" ama {type(value).__name__} tipinde değer verildi.")
        
        print(f"'{self.public_name}' ayarlanıyor, yeni değer: {value}")
        # private name kullanarak instance'ın __dict__'ine değeri yaz
        instance.__dict__[self.private_name] = value


print("\n`Typed` descriptor'ı ile bir class tanımlayalım:")

class Person:
    name = Typed(str)
    age = Typed(int)

    def __init__(self, name: str, age: int):
        # __init__ içindeki atamalar, descriptor'ın __set__ metodunu tetikler
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

print("\nPerson objesi oluşturalım:")
p = Person("Ali", 30)

print("\nAttribute'lara erişelim (descriptor'ın __get__'i tetiklenir):")
print(f"Kişinin adı: {p.name}")
print(f"Kişinin yaşı: {p.age}")

print("\nGeçersiz bir tip ataması yapmayı deneyelim:")
try:
    p.age = "kırk"
except TypeError as e:
    print(f"Hata yakalandı: {e}")

print("\n\n--- 3. Basit Bir ORM Field Simülasyonu ---")
# `Typed` descriptor'ı o kadar güçlü ki, onu doğrudan bir "Field" olarak kullanabiliriz.
class Field(Typed):
    """
    ORM Field'ı gibi davranan bir descriptor.
    Aslında sadece daha anlamlı bir isimle yeniden adlandırılmış Typed descriptor'ı.
    """
    pass

class Product:
    name = Field(str)
    price = Field(float)
    stock = Field(int)
    
    def __init__(self, name: str, price: float, stock: int):
        self.name = name
        self.price = price
        self.stock = stock

    def __repr__(self):
        return f"Product(name='{self.name}', price={self.price}, stock={self.stock})"

print("\nProduct objesi oluşturalım:")
product = Product("Laptop", 25000.50, 15)
print(product)

print("\nBir attribute'u güncelleyelim:")
product.stock = 14
print(product)
