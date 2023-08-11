const client = await newGrpcConnection();

const gateway = connect({
    client,
    identity: await newIdentity(),
    signer: await newSigner(),
});

async function newIdentity(): Promise<Identity> {
    const credentials = await fs.readFile(certPath);
    return { mspId: 'Org1MSP', credentials };
}

async function newSigner(): Promise<Signer> {
    const privateKeyPem = await fs.readFile(keyPath);
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}