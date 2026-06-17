-- Execute no MySQL do Railway (aba Data ou cliente SQL)
CREATE TABLE IF NOT EXISTS user_rec_facial (
    imgID       INT AUTO_INCREMENT PRIMARY KEY,
    imgVetor    BLOB NOT NULL,
    imgNome     VARCHAR(255) NOT NULL,
    imgData     DATETIME DEFAULT CURRENT_TIMESTAMP,
    imgRelacao  VARCHAR(100) DEFAULT 'conhecido',
    imgUser     INT DEFAULT 0,
    imgLabel    VARCHAR(255) DEFAULT NULL,
    avisar      TINYINT(1) DEFAULT 1
);
