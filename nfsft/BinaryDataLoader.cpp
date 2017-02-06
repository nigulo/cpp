#include "BinaryDataLoader.h"

BinaryDataLoader::BinaryDataLoader(const string& fileName, int bufferSize,
		const vector<int>& dims,
		const vector<vector<pair<int, int>>>& regions,
		int totalNumVars, const vector<int>& varIndices, int type, Precision prec) :
		fileName(fileName),
		bufferSize(bufferSize),
		mode(ios::in | ios::binary),
		dims(dims),
		regions(regions),
		totalNumVars(totalNumVars),
		varIndices(varIndices),
		input(fileName, mode),
		page(-1),
		data(nullptr),
		pageSize(0)	,
		type(type),
		sizeOfReal(prec == SinglePrecision ? sizeof(float) : sizeof (double)) {
	assert(!(mode & ios::out)); // Don't allow write mode
	assert(bufferSize > 0);
	assert(input.is_open());
	dim = 1;
	for (auto dimx : dims) {
		dim *= dimx;
	}
	for (int varIndex : varIndices) {
		assert(varIndex < GetTotalNumVars());
	}
	inRegion = new bool[dim];
	for (int i = 0; i < dim; i++) {
		inRegion[i] = InRegion(i);
	}

}

BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) :
		fileName(dataLoader.fileName),
		bufferSize(dataLoader.bufferSize),
		mode(dataLoader.mode),
		dims(dataLoader.dims),
		regions(dataLoader.regions),
		totalNumVars(dataLoader.totalNumVars),
		varIndices(dataLoader.varIndices),
		input(dataLoader.fileName, dataLoader.mode),
		page(-1),
		data(nullptr),
		pageSize(0),
		dim(dataLoader.dim),
		type(dataLoader.type),
		sizeOfReal(dataLoader.sizeOfReal) {

	assert(input.is_open());
	inRegion = new bool[dim];
	for (int i = 0; i < dim; i++) {
		inRegion[i] = dataLoader.inRegion[i];
	}

	if (dataLoader.page >= 0) {
		if (RECORDHEADER) {
			input.seekg((dataLoader.page + 1) * bufferSize * ((dim * totalNumVars + 1) * sizeOfReal + 16), ios::cur);
		} else {
			input.seekg((dataLoader.page + 1) * bufferSize * (dim * totalNumVars + 1) * sizeOfReal, ios::cur);
		}
	}
	page = dataLoader.page;
	Next();
}

BinaryDataLoader::~BinaryDataLoader() {
	if (input.is_open()) {
		input.close();
	}
	if (inRegion) {
		delete[] inRegion;
	}
	if (data) {
		delete[] data;
	}
}

bool BinaryDataLoader::Next() {
	if (data) {
		delete[] data;
		data = nullptr;
	}
	if (!input.is_open()) {
		pageSize = 0;
		return false;
	}
	page++;
	size_t varSize = dim * GetNumVars() + 1;
	size_t dataPageSize = bufferSize * varSize;
	assert(dataPageSize * sizeOfReal < 4294967296 && "Decrease bufferSize"); // 4GB
	data = new char[dataPageSize * sizeOfReal];
	if (RECORDHEADER) {
		assert(sizeof (int) == 4);
		int i = 0;
		int dataOffset = 0;
		while (i < bufferSize) {
			int recordSize;
			input.read((char*) &recordSize, 4);
			int numBytesRead = input.gcount();
			if (input.eof()) {
				input.close();
				break;
			}
			assert(numBytesRead == 4);
			//cout << "dim, totalnumvars, recordsize: " << dim << " " << totalNumVars << " " << recordSize << endl;
			if (type == TYPE_VIDEO) {
				recordSize -= 2 * sizeOfReal; // time and position are included in the block
			}
			assert(recordSize == (sizeOfReal * (dim * totalNumVars)));
			int lastVarIndex = 0;
			bool eof = false;
			for (int varIndex : GetVarIndices()) {
				assert(varIndex >= lastVarIndex);
				if (varIndex - lastVarIndex != 0) {
					input.seekg((varIndex - lastVarIndex) * dim * sizeOfReal, ios::cur);
				}
				input.read(data + dataOffset, dim * sizeOfReal);
				numBytesRead = input.gcount();
				if (lastVarIndex == 0 && numBytesRead == 0) {
					cout << "Premature end of data file reached" << endl;
					input.close();
					eof = true;
					break;
				}
				//cout << "numBytesRead: " << numBytesRead << endl;
				assert(numBytesRead == dim * sizeOfReal);
				lastVarIndex = varIndex + 1;
				dataOffset += dim * sizeOfReal;
			}
			if (eof) {
				break;
			}
			assert(recordSize >= lastVarIndex * dim * sizeOfReal);
			if (recordSize - lastVarIndex * dim * sizeOfReal != 0) {
				input.seekg(recordSize - lastVarIndex * dim * sizeOfReal, ios::cur);
			}
			if (type == TYPE_SNAPSHOT) {
				input.read((char*) &recordSize, 4);
				numBytesRead = input.gcount();
				assert(numBytesRead == 4);
				input.read((char*) &recordSize, 4);
				numBytesRead = input.gcount();
				assert(numBytesRead == 4);
				//cout << "recordsize: " << recordSize << endl;

				assert(recordSize == 1712);//792);
				input.read(data + dataOffset, sizeOfReal); // time
				dataOffset += sizeOfReal;
				numBytesRead = input.gcount();
				assert(numBytesRead == sizeOfReal);
				input.seekg((recordSize - 1) * sizeOfReal, ios::cur);
				//input.read((char*) &recordSize, 4);
				//numBytesRead = input.gcount();
				//assert(numBytesRead == 4);
			} else {
				input.read(data + dataOffset, sizeOfReal); // time
				dataOffset += sizeOfReal;
				numBytesRead = input.gcount();
				assert(numBytesRead == sizeOfReal);
				input.seekg(sizeOfReal, ios::cur); // skip position
				input.read((char*) &recordSize, 4);
				numBytesRead = input.gcount();
				assert(numBytesRead == 4);
			}
			i++;
		}
		pageSize = i;
	} else {
		input.read(data, (sizeOfReal) * dataPageSize);
		int numBytesRead = input.gcount();
		if (numBytesRead < (sizeOfReal) * dataPageSize) {
			assert(numBytesRead % ((sizeOfReal) * varSize) == 0);
			pageSize = numBytesRead / ((sizeOfReal) * varSize);
			input.close();
		} else {
			pageSize = bufferSize;
		}
	}
	return pageSize > 0;
}

BinaryDataLoader* BinaryDataLoader:: Clone() const {
	BinaryDataLoader* dl = new BinaryDataLoader(*this);
	if (dl->GetPageSize() == 0) {
		delete dl;
		return nullptr;
	}
	return dl;
}
