# خطوات رفع كود البحث على GitHub للمجلة

1. افتح https://github.com/new وسجّل الدخول.
2. اسم المستودع المقترح: `lung-ct-spatial-googlenet-pca-svm`.
3. اختر **Public** فقط بعد التأكد من حقوق مشاركة ملفات النتائج والبحث.
4. اضغط **Create repository**.
5. لأن الحزمة تضم أكثر من 100 ملف، **لا ترفعها دفعة واحدة عبر صفحة المتصفح**؛ GitHub يحدّد 100 ملف لكل عملية رفع من المتصفح. استخدم GitHub Desktop أو git لرفع كل المجلد بسهولة.
6. في **GitHub Desktop** اختر `File → Add local repository` وحدد المجلد المفكوك. إذا لم يكن مستودع Git بعد، اختر `Create a Repository` لهذا المجلد، ثم `Publish repository`.
7. افتح تبويب **Actions**، وتأكد أن `Research-code smoke tests` يعمل بنجاح.
8. انسخ رابط المستودع (مثال: `https://github.com/YOUR_ACCOUNT/lung-ct-spatial-googlenet-pca-svm`) وضعه في Data/Code Availability داخل البحث.
9. بعد تثبيت النسخة النهائية، أنشئ **Release** في GitHub ويمكن ربطه بـZenodo للحصول على DOI حقيقي.

## رفع محتويات المجلد بـGit (أفضل من رفع ZIP فقط)

```bash
cd LungCT_Spatial_Features_GitHub
git init
git add .
git commit -m "Release reproducible research code and audited results"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/lung-ct-spatial-googlenet-pca-svm.git
git push -u origin main
```

**تنبيه:** لا ترفع صور المرضى الأصلية، ملفات DICOM، كلمة مرور، توكن GitHub، أو ملفات feature cache الضخمة بغير إذن. لا تحذف النتائج الخارجية الضعيفة؛ وثّقها على حقيقتها.

### نص مقترح للمقالة (بعد نشر المستودع فعلاً)

> **Code Availability:** The Python source code, executable Kaggle notebooks, experimental configuration files, and audited prediction-level results supporting this study are publicly available at [INSERT ACTUAL GITHUB REPOSITORY URL]. The repository documents preprocessing, similarity-grouped cross-validation, nested model selection, and exploratory external evaluations. The original CT image datasets and proprietary/third-party assets are not redistributed; dataset access instructions are included in the repository.

لا تستخدم هذا النص مع رابط وهمي قبل إنشاء المستودع فعلياً.
