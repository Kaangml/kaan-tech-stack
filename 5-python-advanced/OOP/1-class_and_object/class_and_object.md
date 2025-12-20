# 1. Python'da Sınıflar ve Nesneler: Derinlemesine Bir Bakış

Bu bölüm, Python'un nesne yönelimli programlama (OOP) yeteneklerinin temellerinin ötesine geçerek, nesne modelinin (Python Object Model) altında yatan güçlü mekanizmaları incelemektedir. Sadece `class` anahtar kelimesini kullanmakla kalmayıp, bir nesnenin ve hatta bir sınıfın yaşam döngüsünün her adımına nasıl müdahale edebileceğimizi keşfedeceğiz.

Burada ele alınan konular, özellikle esnek ve güçlü framework'ler (Django, Pydantic, PyTorch vb.) geliştirmek veya bu framework'lerin "sihrinin" arkasında ne olduğunu anlamak için kritik öneme sahiptir.

## İncelenen Konular

### [1. Nesneler (Objects)](./1-objects/objects.md)
Bir nesnenin hayata geliş anını kontrol eden `__new__` ve `__init__` metotları arasındaki temel fark nedir? Nesnelerimizi insan dostu (`__str__`) ve geliştirici dostu (`__repr__`) formatlarda nasıl temsil ederiz? Bu bölümde, bir nesnenin yaşam döngüsünün temellerini sağlamlaştırıyoruz.

### [2. Descriptor Protokolü](./2-descriptor/descriptor.md)
Python'un en güçlü ama en az anlaşılan özelliklerinden biri. Descriptor'lar, bir nesnenin attribute'larına erişildiği, değer atandığı veya silindiği anları "yakalamamızı" (intercept) sağlar. `@property`, `@staticmethod` gibi yapıların arkasındaki mekanizmayı ve kendi "akıllı" attribute'larımızı nasıl yazabileceğimizi bu bölümde öğreniyoruz.

### [3. Metaclasses](./3-metaclass/metaclass.md)
Eğer descriptor'lar attribute erişimini kontrol ediyorsa, metaclass'ler de **sınıfların yaratılışını** kontrol eder. "Sınıfların sınıfı" olarak bilinen bu konsept, bir sınıf tanımlandığı anda ona müdahale etme, onu otomatik olarak bir yere kaydetme (registry pattern) veya yapısını tamamen değiştirme gücü verir. Framework'lerin temel yapı taşıdır.

### [4. Dataclasses & Attrs](./4-dataclasses_attrs/dataclasses_attrs.md)
Özellikle veri saklamak için kullanılan sınıflardaki `__init__`, `__repr__`, `__eq__` gibi "boilerplate" (tekrar eden standart) kodları otomatik olarak üreten bu modern Python araçlarını inceliyoruz. Daha az kodla daha sağlam ve okunaklı sınıflar yazmanın yolları.