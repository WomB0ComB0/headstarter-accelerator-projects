import { RequestListener } from 'node:http';
import { L as ListenOptions, a as Listener, H as HTTPSOptions, C as Certificate } from './shared/listhen.994822d6.js';
export { G as GetURLOptions, b as ListenURL, S as ShowURLOptions } from './shared/listhen.994822d6.js';
import { ConsolaInstance } from 'consola';
import * as h3 from 'h3';
import forge from 'node-forge';
import 'node:https';
import 'node:net';
import 'get-port-please';

declare function listen(handle: RequestListener, _options?: Partial<ListenOptions>): Promise<Listener>;

interface DevServerOptions {
    cwd?: string;
    staticDirs?: string[];
    logger?: ConsolaInstance;
}
declare function createDevServer(entry: string, options: DevServerOptions): Promise<{
    cwd: string;
    resolver: {
        relative: (path: string) => string;
        formateRelative: (path: string) => string;
        import: (id: string) => Promise<any>;
        resolve: (id: string) => string;
        tryResolve: (id: string) => string | undefined;
    };
    nodeListener: h3.NodeListener;
    reload: (_initial?: boolean) => Promise<void>;
    _entry: string | undefined;
}>;

interface WatchOptions extends DevServerOptions {
    cwd?: string;
    logger?: ConsolaInstance;
    ignore?: string[];
    publicDirs?: string[];
}
declare function listenAndWatch(entry: string, options: Partial<ListenOptions & WatchOptions>): Promise<Listener>;

interface CertificateOptions {
    validityDays: number;
    subject: forge.pki.CertificateField[];
    issuer: forge.pki.CertificateField[];
    extensions: any[];
}
interface CommonCertificateOptions {
    commonName?: string;
    countryCode?: string;
    state?: string;
    locality?: string;
    organization?: string;
    organizationalUnit?: string;
    emailAddress?: string;
    domains?: string[];
}
interface SigningOptions {
    signingKey?: string;
    signingKeyCert?: string;
    signingKeyPassphrase?: string;
}
interface TLSCertOptions extends CommonCertificateOptions, SigningOptions {
    bits?: number;
    validityDays?: number;
    passphrase?: string;
}
declare function resolveCertificate(options: HTTPSOptions): Promise<Certificate>;
declare function generateCertificates(options: TLSCertOptions): Promise<{
    cert: Certificate;
    ca: Certificate;
}>;
declare function resolveCert(options: HTTPSOptions): Promise<Certificate>;
declare function resolvePfx(options: HTTPSOptions): Promise<forge.pkcs12.Pkcs12Pfx>;
declare function createCertificateInfo(options: CommonCertificateOptions): {
    attributes: {
        name: string;
        value: string;
    }[];
    extensions: ({
        name: string;
        cA: boolean;
        critical: boolean;
        digitalSignature?: undefined;
        keyEncipherment?: undefined;
        serverAuth?: undefined;
        clientAuth?: undefined;
        altNames?: undefined;
    } | {
        name: string;
        digitalSignature: boolean;
        keyEncipherment: boolean;
        critical: boolean;
        cA?: undefined;
        serverAuth?: undefined;
        clientAuth?: undefined;
        altNames?: undefined;
    } | {
        name: string;
        serverAuth: boolean;
        clientAuth: boolean;
        cA?: undefined;
        critical?: undefined;
        digitalSignature?: undefined;
        keyEncipherment?: undefined;
        altNames?: undefined;
    } | {
        name: string;
        altNames: ({
            type: number;
            ip: string;
            value?: undefined;
        } | {
            type: number;
            value: string;
            ip?: undefined;
        })[];
        cA?: undefined;
        critical?: undefined;
        digitalSignature?: undefined;
        keyEncipherment?: undefined;
        serverAuth?: undefined;
        clientAuth?: undefined;
    })[];
};
declare function createCaInfo(options: TLSCertOptions): {
    attributes: {
        name: string;
        value: string;
    }[];
    extensions: ({
        name: string;
        cA: boolean;
        critical: boolean;
        keyCertSign?: undefined;
    } | {
        name: string;
        keyCertSign: boolean;
        critical: boolean;
        cA?: undefined;
    })[];
};
declare function generateTLSCert(options: TLSCertOptions): Promise<Certificate>;
declare function generateCACert(options?: TLSCertOptions): Promise<Certificate>;
declare function signCertificate(options: SigningOptions, cert: forge.pki.Certificate): void;
declare function createCertificateFromKeyPair(keyPair: forge.pki.KeyPair, options: CertificateOptions): forge.pki.Certificate;
declare function generateKeyPair(bits?: number): Promise<forge.pki.KeyPair>;
declare function generateCert(options: TLSCertOptions & CertificateOptions): Promise<Certificate>;
declare const _private: {
    generateCert: typeof generateCert;
    generateKeyPair: typeof generateKeyPair;
    generateCACert: typeof generateCACert;
    generateTLSCert: typeof generateTLSCert;
    createCaInfo: typeof createCaInfo;
    createCertificateInfo: typeof createCertificateInfo;
    signCertificate: typeof signCertificate;
    createCertificateFromKeyPair: typeof createCertificateFromKeyPair;
    resolveCert: typeof resolveCert;
    resolvePfx: typeof resolvePfx;
    generateCertificates: typeof generateCertificates;
};

export { Certificate, type CertificateOptions, type CommonCertificateOptions, type DevServerOptions, HTTPSOptions, ListenOptions, Listener, type SigningOptions, type TLSCertOptions, type WatchOptions, _private, createDevServer, listen, listenAndWatch, resolveCertificate };
