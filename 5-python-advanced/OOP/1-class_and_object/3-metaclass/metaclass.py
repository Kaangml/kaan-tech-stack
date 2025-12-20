# -*- coding: utf-8 -*- 

"""
Bu dosya, metaclass.md dosyasında anlatılan Metaclass konseptini ve
pratik bir "Registry" (Kayıt Sistemi) desenini kod örnekleriyle gösterir.
"""

print("--- 1. Metaclass'in Temelleri ---")

# Adım 1: Custom Metaclass'i Tanımla
# Metaclass'ler neredeyse her zaman 'type' sınıfından miras alır.
class MyMeta(type):
    # Metaclass'in __new__ metodu, bir sınıf yaratılmadan hemen önce çalışır.
    # mcs: Metaclass'in kendisi (MyMeta)
    # name: Yaratılacak sınıfın adı (örneğin, "MySpam")
    # bases: Sınıfın miras alacağı parent class'ların tuple'ı
    # attrs: Sınıfın attribute ve metotlarını içeren dictionary
    def __new__(mcs, name, bases, attrs):
        print(f"METACLASS __new__ ÇAĞRILDI:")
        print(f"  mcs         = {mcs}")
        print(f"  name        = '{name}'")
        print(f"  bases       = {bases}")
        print(f"  attrs       = {list(attrs.keys())}")
        
        # Sınıfın attribute'larına müdahale edebiliriz.
        # Örneğin, her sınıfa otomatik olarak bir 'author' attribute'u ekleyelim.
        attrs['author'] = "Kaan"
        
        # Yeni sınıf nesnesini yaratmak için üst sınıfın __new__ metodunu çağırmalıyız.
        new_class = super().__new__(mcs, name, bases, attrs)
        
        print(f"  -> '{name}' sınıfı yaratıldı ve döndürülüyor.\n")
        return new_class

    # Metaclass'in __init__ metodu, sınıf yaratıldıktan sonra çalışır.
    # cls: Yeni yaratılmış olan sınıfın kendisi
    def __init__(cls, name, bases, attrs):
        print(f"METACLASS __init__ ÇAĞRILDI:")
        print(f"  cls         = {cls}")
        print(f"  name        = '{name}'")
        print("  -> Yaratılan sınıf üzerinde ek işlemler yapılabilir.\n")
        super().__init__(name, bases, attrs)


# Adım 2: Metaclass'i Kullan
# Python 3 sözdizimi ile metaclass'i belirtiyoruz.
class MySpam(metaclass=MyMeta):
    ham = 42
    
    def eggs(self):
        return self.ham

print("MySpam sınıfı tanımlandı. Şimdi attribute'larını kontrol edelim:")
print(f"MySpam.ham = {MySpam.ham}")
# MyMeta.__new__ tarafından eklenen attribute:
print(f"MySpam.author = '{MySpam.author}'")


print("\n\n--- 2. Pratik Örnek: Otomatik Plugin Kayıt Sistemi ---")

class PluginMeta(type):
    """
    Bu metaclass, kendisini kullanan Base sınıfından türeyen tüm alt sınıfları
    otomatik olarak bir kayıt defterine (registry) ekler.
    """
    # Kayıt defterini metaclass'in bir attribute'u olarak tutuyoruz.
    _plugins = {}

    def __init__(cls, name, bases, attrs):
        # Bu __init__ metodu, her yeni plugin sınıfı tanımlandığında çalışır.
        # cls -> yeni yaratılan sınıf (örn: JSONParser, XMLParser)
        print(f"PLUGIN_META: '{name}' sınıfı tanımlanıyor...")
        
        # Base sınıfın kendisini kaydetmeyi atla, sadece alt sınıfları kaydet.
        if name != 'BasePlugin':
            # Sınıfın ismini küçük harflerle anahtar olarak kullanalım.
            plugin_id = name.lower()
            if plugin_id in cls._plugins:
                raise TypeError(f"'{plugin_id}' id'li plugin zaten mevcut!")

            # Yeni sınıfı kayıt defterine ekle.
            cls._plugins[plugin_id] = cls
            print(f"  -> '{name}' plugini '{plugin_id}' id ile kaydedildi.")

        # Metaclass'in __init__'i içinde super().__init__ çağırmak iyi bir pratiktir.
        super().__init__(name, bases, attrs)

# Ana 'BasePlugin' sınıfımız bu metaclass'i kullanır.
# Bu sayede, bu sınıftan türeyen HERKES otomatik olarak PluginMeta'dan geçer.
class BasePlugin(metaclass=PluginMeta):
    def parse(self, data):
        raise NotImplementedError("Her plugin kendi parse metodunu yazmalıdır!")

# Artık geliştiriciler sadece BasePlugin'den miras alarak yeni pluginler yazabilir.
# Hiçbir yerde register() gibi bir fonksiyon çağırmalarına gerek yok.

print("\nPluginler tanımlanıyor...")

class JSONParser(BasePlugin):
    def parse(self, data: str):
        print(f"JSONParser: '{data}' verisi ayrıştırılıyor.")
        return {"json_data": data}

class XMLParser(BasePlugin):
    def parse(self, data: str):
        print(f"XMLParser: '{data}' verisi ayrıştırılıyor.")
        return f"<xml>{data}</xml>"

class YAMLParser(BasePlugin):
    def parse(self, data: str):
        print(f"YAMLParser: '{data}' verisi ayrıştırılıyor.")
        return f"yaml: {data}"

print("\nTüm pluginler tanımlandı!")

# Kayıt defterini kontrol edelim
print("\nMevcut pluginler:")
print(PluginMeta._plugins)

# Bu sistemi kullanarak dinamik olarak bir parser seçip kullanabiliriz.
def process_data(data: str, format_id: str):
    format_id = format_id.lower()
    
    # Kayıt defterinden doğru parser sınıfını bul
    parser_class = PluginMeta._plugins.get(format_id)
    
    if not parser_class:
        raise ValueError(f"'{format_id}' formatı için uygun bir parser bulunamadı!")
        
    # Parser'dan bir nesne yarat ve kullan
    parser_instance = parser_class()
    return parser_instance.parse(data)

print("\nPlugin sistemini test edelim:")
process_data("Merhaba Dünya", "jsonparser")
process_data("<html>...</html>", "xmlparser")

try:
    process_data("test", "csvparser")
except ValueError as e:
    print(f"\nBeklenen hata yakalandı: {e}")
