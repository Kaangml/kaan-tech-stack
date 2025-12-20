# -*- coding: utf-8 -*- 

"""
Bu dosya, dataclasses_attrs.md dosyasında anlatılan konseptleri
somut kod örnekleriyle göstermek için hazırlanmıştır.
"""

from dataclasses import dataclass, field, FrozenInstanceError

print("---""- 1. Sorun: 'Boilerplate' Kod Cehennemi ---""" ) 

class ClassicPoint:
    """Sadece __init__ yazıldığında karşılaşılan sorunları gösteren sınıf."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1_classic_bad = ClassicPoint(1, 2)
p2_classic_bad = ClassicPoint(1, 2)

print(f"ClassicPoint'in kötü __repr__'ı: {p1_classic_bad}")
print(f"Değerleri aynı olmasına rağmen karşılaştırma sonucu: {p1_classic_bad == p2_classic_bad}")


class FullClassicPoint:
    """Tüm dunder metotlarının elle yazıldığı 'geveze' sınıf."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"FullClassicPoint(x={self.x}, y={self.y})"

    def __eq__(self, other):
        if not isinstance(other, FullClassicPoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

print("\nTüm dunder'lar elle yazılınca:")
p1_classic_good = FullClassicPoint(1, 2)
p2_classic_good = FullClassicPoint(1, 2)
print(f"Elle yazılmış __repr__: {p1_classic_good}")
print(f"Elle yazılmış __eq__ ile karşılaştırma: {p1_classic_good == p2_classic_good}")


print("\n\n---""- 2. Çözüm: @dataclass ---""" )

@dataclass
class Point:
    """Aynı sınıfın @dataclass ile modern ve kısa hali."""
    x: int
    y: int

p1_dc = Point(1, 2)
p2_dc = Point(1, 2)

print(f"@dataclass'in otomatik ürettiği __repr__: {p1_dc}")
print(f"@dataclass'in otomatik ürettiği __eq__ ile karşılaştırma: {p1_dc == p2_dc}")


print("\n\n---""- 3. Gelişmiş @dataclass Özellikleri ---""" )

@dataclass(frozen=True, order=True)
class SecureUser:
    """
    Değiştirilemez (frozen) ve sıralanabilir (order) bir dataclass örneği.
    Ayrıca bazı field'lar özelleştirilmiştir.
    """
    username: str
    # Düzeltme: `compare=False` eklendi. Artık karşılaştırmalar sadece `username` üzerinden yapılacak.
    password: str = field(repr=False, compare=False)
    roles: list[str] = field(default_factory=list, compare=False)

user1 = SecureUser("kaan", "12345", roles=['admin', 'user'])
user2 = SecureUser("ahmet", "abcde")
user3 = SecureUser("kaan", "xxxxx") # username aynı, password farklı

print(f"\nrepr'da parola görünmüyor: {user1}")

print("\n'frozen=True' testi:")
try:
    user1.username = "yeni_kaan"
except FrozenInstanceError as e:
    print(f"Değiştirme engellendi, beklenen hata: {e}")

# `order=True` olduğu ve diğer tüm alanlarda `compare=False` olduğu için,
# sıralama sadece `username` alanına göre yapılır.
print("\n'order=True' testi:")
print(f"user1 > user2 mi? ('kaan' > 'ahmet'): {user1 > user2}")
print(f"user1 < user2 mi? ('kaan' < 'ahmet'): {user1 < user2}")

# AÇIKLAMA:
# Eşitlik karşılaştırması (`__eq__`), `compare=True` olan alanların (bu örnekte sadece `username`)
# bir `tuple` içine alınıp bu `tuple`'ların karşılaştırılmasıyla çalışır.
# Kodumuz `(self.username,) == (other.username,)` şeklinde bir karşılaştırma yapar.
# `password` ve `roles` alanları `compare=False` olduğu için bu `tuple`'a dahil edilmez.
# Bu nedenle, `user1` ve `user3`'ün `username`'leri aynı olduğu için sonuç `True` olacaktır.
print(f"\nuser1 == user3 mü? ('kaan' == 'kaan'): {user1 == user3}")


print("\n'default_factory' testi:")
user4 = SecureUser("ayse", "secret")
user5 = SecureUser("fatma", "pass")
print(f"user4 rolleri: {user4.roles}")
user4.roles.append("guest")
print(f"user4 rolleri güncellendi: {user4.roles}")
print(f"user5 rolleri etkilenmedi: {user5.roles}")


# print("\n\n---""- 4. Alternatif: attrs Kütüphanesi ---""" )
# # Bu bölümü çalıştırmak için: pip install attrs
# try:
#     import attr

#     @attr.s(auto_attribs=True)
#     class AttrsPoint:
#         x: int
#         y: int
    
#     p_attrs = AttrsPoint(1, 2)
#     print(f"\nattrs ile üretilen __repr__: {p_attrs}")
#     print(f"attrs ile karşılaştırma: {p_attrs == AttrsPoint(1, 2)}")

# except ImportError:
#     print("\n'attrs' kütüphanesi yüklü değil. 'pip install attrs' ile yükleyebilirsiniz.")
