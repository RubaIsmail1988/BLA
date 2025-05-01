<?php
// الاتصال بقاعدة البيانات
$servername = "localhost";
$username_db = "root"; // اسم المستخدم لقاعدة البيانات
$password_db = ""; // كلمة المرور لقاعدة البيانات
$dbname = "users_db"; // اسم قاعدة البيانات

// إنشاء الاتصال بقاعدة البيانات
$conn = new mysqli($servername, $username_db, $password_db, $dbname);

// التحقق من الاتصال
if ($conn->connect_error) {
    die("الاتصال بقاعدة البيانات فشل: " . $conn->connect_error);
}

// التعامل مع البيانات المدخلة من النموذج
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // استعلام لإدخال البيانات في قاعدة البيانات
    $sql = "INSERT INTO users (username, password) VALUES (?, ?)";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $username, $password);

    // تنفيذ الاستعلام
    if ($stmt->execute()) {
        // إعادة التوجيه إلى صفحة success.php
        header("Location: success.php");
        exit();
    } else {
        // في حال فشل الإدخال
        echo "خطأ في التسجيل: " . $stmt->error;
    }

    // إغلاق الاتصال
    $stmt->close();
}

$conn->close();
?>
